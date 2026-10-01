"""About, and the manual.

Two dialogs and the text they show. Both are plain HTML in a ``QTextBrowser``,
which is the same shape LARMOR and PRISM use, so someone who has met one of
those knows where to look here.

The manual is written into this module rather than read from a file next to the
executable. That is deliberate: a frozen application's working directory is
wherever the user launched it from, a file beside the ``.exe`` can be deleted by
a well-meaning person tidying up a shared folder, and a manual that sometimes
fails to open is worse than a shorter one that always does. Being a Python
module also means the version number, the tagline and the default thresholds in
the text come from the code rather than from a copy of it that can drift.

The voice is the application's: it says what a number is and where it came from,
and it leaves the judgement to the reader.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import bv
from ..version import NAME, TAGLINE, __version__
from . import chrome


def _offset(v: float) -> float:
    """How far beyond R0 a threshold of ``v`` reaches: -b ln v."""
    return -bv.DEFAULT.b * math.log(v)


# ---------------------------------------------------------------------------
# about
# ---------------------------------------------------------------------------

def about_html(renderer: str = "", theme=None) -> str:
    """The About text. ``renderer`` is the tier actually in use, if known."""
    c = chrome.ui_colors(theme)
    d_bond = _offset(bv.V_BOND_DEFAULT)
    return f"""
<h2 style="margin-bottom:2px">{NAME}
  <span style="font-size:14px;color:{c.muted};font-weight:normal">
  v{__version__}</span></h2>
<p style="color:{c.muted};margin-top:0"><i>{TAGLINE}.</i></p>

<p>Written by <b>Sam Soudani</b>, in the <b>McCloy</b> group at
<b>Washington State University</b>.</p>
<p><a href="https://github.com/sams808/FACET">github.com/sams808/FACET</a>
&nbsp;·&nbsp; MIT&nbsp;Licence</p>

<h3 style="margin-bottom:2px">What it is for</h3>
<p style="margin-top:2px">A published coordination number is a function of the
distance cutoff that produced it, and that cutoff is rarely stated. FACET
reports the number together with the threshold it came from, and sets the
threshold in bond valence rather than in distance:</p>
<p align="center"><i>v</i> = exp((<i>R</i><sub>0</sub> &minus; <i>d</i>) /
<i>b</i>)&nbsp;,&nbsp;&nbsp; <i>b</i> = {bv.DEFAULT.b} &Aring;</p>
<p style="margin-top:2px">One valence threshold therefore becomes a different
distance for every element pair, through that pair's own
<i>R</i><sub>0</sub>. The shipped defaults are
<b>{bv.V_BOND_DEFAULT:g}&nbsp;v.u.</b> for a bond
(<i>d</i> = <i>R</i><sub>0</sub> + {d_bond:.3f}&nbsp;&Aring;) and
<b>{bv.V_LIST_DEFAULT:g}&nbsp;v.u.</b> for the contact table.</p>

<h3 style="margin-bottom:2px">This session</h3>
<table cellpadding="2" style="margin-top:2px">
  <tr><td style="color:{c.muted}">Renderer</td>
      <td>{renderer or 'not initialised'}</td></tr>
  <tr><td style="color:{c.muted}">Bond-valence parameters</td>
      <td>{bv.DEFAULT.name}</td></tr>
  <tr><td style="color:{c.muted}">Parameter source</td>
      <td>{bv.DEFAULT.source}</td></tr>
</table>
<p style="color:{c.muted};font-size:11px">Every file FACET writes carries these
same lines, plus the two thresholds in use, so a table can be traced back to the
settings that produced it.</p>

<h3 style="margin-bottom:2px">Built on</h3>
<ul style="margin-top:2px">
  <li><b>gemmi</b> — crystallographic file handling and symmetry
      (Wojdyr, <i>J. Open Source Softw.</i> <b>7</b>, 4200, 2022)</li>
  <li><b>spglib</b> — space-group determination
      (Togo &amp; Tanaka, arXiv:1808.01590)</li>
  <li><b>NumPy</b> — Harris et al., <i>Nature</i> <b>585</b>, 357 (2020)</li>
  <li><b>SciPy</b> — Virtanen et al., <i>Nature Methods</i> <b>17</b>, 261
      (2020)</li>
  <li><b>PySide6 / Qt for Python</b> (The Qt Company) — the interface and the
      OpenGL renderer</li>
  <li><b>pymatgen</b> — optional, for the alternative near-neighbour rules
      (Ong et al., <i>Comput. Mater. Sci.</i> <b>68</b>, 314, 2013)</li>
</ul>

<h3 style="margin-bottom:2px">Bond-valence parameters</h3>
<p style="margin-top:2px">The built-in set is small and is stated as such.
FACET reads the published compilations from a file you supply — the IUCr
<code>bvparm</code> distribution, a softBV-style table, or a FACET parameter
file — under <i>File&nbsp;&rsaquo;&nbsp;Load bond-valence parameters</i>. Those
compilations are not bundled, because each arrives with its own terms of use.
Cite whichever set your numbers came from; the name is in every export.</p>

<h3 style="margin-bottom:2px">Inspired by</h3>
<ul style="margin-top:2px">
  <li><b>VESTA</b> — Momma &amp; Izumi, <i>J. Appl. Crystallogr.</i>
      <b>44</b>, 1272 (2011)</li>
  <li><b>CrystalMaker</b> — CrystalMaker Software Ltd</li>
  <li><b>Brown's bond-valence model</b> — I. D. Brown,
      <i>The Chemical Bond in Inorganic Chemistry</i>, IUCr/OUP (2002)</li>
</ul>

<p style="font-size:8pt;color:{c.muted}">&copy; 2026 NOME Group, Washington
State University. FACET's own source is MIT licensed. The built application
bundles Qt under the LGPL-3.0 — see <i>Help &rsaquo; Licences</i> for what that
allows and requires when passing a build on.</p>
"""


# ---------------------------------------------------------------------------
# the manual
# ---------------------------------------------------------------------------
# Each entry is (key, title, body). The key is what other code addresses a
# section by, so it is part of the interface and does not change with the title.

def _sections(theme=None) -> list[tuple[str, str, str]]:
    c = chrome.ui_colors(theme)
    mut = c.muted
    note = f"style=\"color:{mut};font-size:11px\""
    d_bond, d_list = _offset(bv.V_BOND_DEFAULT), _offset(bv.V_LIST_DEFAULT)

    return [
        ("start", "Getting started", f"""
<h1>Getting started</h1>
<p>FACET opens a crystal structure, works out what is bonded to what, and shows
how that answer depends on where the line between bonded and not-bonded is
drawn.</p>
<ol>
  <li><b>Open a structure.</b> <i>File &rsaquo; Open structure</i>, or drag a
  file onto the window. CIF, mmCIF, VASP POSCAR/CONTCAR, XYZ, SHELX
  <code>.res</code>/<code>.ins</code>, PDB, VESTA and CrystalMaker files are
  read. <i>File &rsaquo; Open a folder of CIFs</i> loads many at once; the
  <b>Structures</b> dock on the left switches between them and can show several
  overlaid.</li>
  <li><b>Pick a site.</b> Click an atom in the 3D view, or choose a row in the
  <b>Sites</b> dock. The <b>Site</b> tab on the right then describes that
  site.</li>
  <li><b>Move the threshold.</b> The strip along the bottom is the cutoff
  explorer. Drag it and the coordination number, the bond-valence sum, the
  contact table and the drawn bonds all follow, with no recalculation of the
  neighbour search.</li>
</ol>
<p {note}>Nothing is loaded from the internet, and nothing is written anywhere
until you export it.</p>
"""),

        ("why", "Why valence, not distance", f"""
<h1>Why valence, not distance</h1>
<p>A coordination number is not measured; it is produced by a rule. The commonest
rule is a distance cutoff — "everything within 3.0 &Aring;" — and the number it
gives changes when the cutoff does. Two papers on the same compound can report
different coordination numbers without either having made an arithmetic
error.</p>
<p>FACET therefore cuts on <b>partial bond valence</b> instead:</p>
<p align="center"><i>v</i> = exp((<i>R</i><sub>0</sub> &minus; <i>d</i>) /
<i>b</i>)</p>
<p>with <i>b</i> = {bv.DEFAULT.b} &Aring; and <i>R</i><sub>0</sub> a tabulated
constant for the element pair and oxidation states. Inverting it,</p>
<p align="center"><i>d</i> = <i>R</i><sub>0</sub> &minus; <i>b</i>&thinsp;ln
<i>v</i></p>
<p>so one valence threshold gives a <i>different</i> distance for every pair. At
the shipped {bv.V_BOND_DEFAULT:g}&nbsp;v.u. a contact counts as a bond out to
<i>R</i><sub>0</sub>&nbsp;+&nbsp;{d_bond:.3f}&nbsp;&Aring;; at the
{bv.V_LIST_DEFAULT:g}&nbsp;v.u. tabulation threshold it is listed out to
<i>R</i><sub>0</sub>&nbsp;+&nbsp;{d_list:.3f}&nbsp;&Aring;. A short Si&ndash;O
bond and a long Bi&ndash;I bond are then being judged by the same physical
criterion rather than by the same number of &aring;ngstr&ouml;ms. The
<b>Cutoffs</b> tab under <b>Tools</b> prints the distance each pair gets, so
the translation is visible rather than implied.</p>
<h2>What FACET does not do with this</h2>
<p>It does not tell you which threshold to use, and it does not mark a
coordination number as the right one. What it reports instead is the
<b>plateau</b>: the range of threshold over which a given count holds. A count
that survives a wide range of threshold and one that changes if the threshold
moves in the fourth decimal place are different situations, and the width is the
statement of which you have.</p>
<p {note}>The uncertainty on <i>R</i><sub>0</sub> is systematic: it scales a
whole bond-valence sum rather than averaging out. FACET propagates it and prints
it beside the sum.</p>
"""),

        ("window", "The workspace", f"""
<h1>The workspace</h1>
<h2>Left</h2>
<p><b>Structures</b> — every file loaded in this session. Tick to show, use the
overlay toggle to draw several in one view, and the cross to remove one.<br>
<b>Sites</b> — the crystallographic sites of the active structure. Selecting one
drives the Site tab, the polyhedron and the cutoff explorer.</p>
<h2>Centre</h2>
<p>The 3D view, with two rows of controls above it: drawing style, polyhedra,
how many cells to tile, the unit cell, the projection, what to label atoms and
bonds with, and the tabulation threshold.</p>
<h2>Right</h2>
<p>Ten tabs. <b>Site</b> is the analysis of the selected site.
<b>Tools</b> holds the whole-structure tables. <b>Diffraction</b> computes a
powder pattern, <b>PDF</b> a pair distribution function, and <b>EXAFS</b> the
shell list and resolution report. <b>Planes</b> cuts lattice planes and slabs.
<b>Styles</b> lists per-atom and per-site styling. <b>Volume</b> maps
bond-valence sum through space. <b>Disorder</b> handles partial occupancy and
alternative configurations. <b>Theme</b> is colour, sizing and palettes.</p>

<p {note}>Most controls state their own name inside themselves rather than
beside them -- a spin box reading "r to 20.0 &Aring;" is the r limit. Hovering
any of them explains what it does, and the settings that are set once and left
sit behind a disclosure arrow.</p>
<h2>Bottom</h2>
<p>The <b>cutoff explorer</b>. Each site is a staircase: coordination number
against threshold, with the flat treads being the plateaux. Drag the vertical
line to move the bond threshold; everything above follows it.</p>
<p {note}>Both splitters can be dragged, and the docks can be undocked or
closed. The window remembers nothing between runs, so a layout you liked has to
be set up again — that is a limitation, not a preference.</p>
"""),

        ("site", "Reading the Site tab", f"""
<h1>Reading the Site tab</h1>
<p>Everything on this tab is a measurement of the selected site at the current
threshold.</p>
<table cellpadding="3" width="100%">
<tr><td width="35%"><b>Coordination number</b></td>
    <td>Contacts whose partial valence exceeds the bond threshold.</td></tr>
<tr><td><b>Bond-valence sum</b></td>
    <td>&Sigma;<i>v</i> over those contacts, occupancy-weighted, with the
    propagated <i>R</i><sub>0</sub> uncertainty and the difference from the
    site's nominal oxidation state.</td></tr>
<tr><td><b>ECoN (Hoppe)</b></td>
    <td>Effective coordination number: a weighted count that needs no cutoff at
    all, shown for comparison with the thresholded one.</td></tr>
<tr><td><b>Maximum-gap split</b></td>
    <td>The count you get by cutting at the largest relative gap in the sorted
    distance list. Where no gap is larger than the tolerance, the full contact
    list is reported rather than an arbitrary split.</td></tr>
<tr><td><b>&phi;</b></td>
    <td>|&Sigma;<i>v</i><sub>i</sub><b>&ucirc;</b><sub>i</sub>| /
    &Sigma;<i>v</i><sub>i</sub> — the length of the bond-valence vector sum
    divided by the scalar sum. Zero for a centrosymmetric environment, and
    larger when the bonding is one-sided. It is dimensionless and exactly
    invariant to an error in <i>R</i><sub>0</sub>, because a common factor
    cancels between numerator and denominator.</td></tr>
<tr><td><b>Void cone half-angle</b></td>
    <td>The widest cone about the site that contains no bonded ligand
    direction.</td></tr>
<tr><td><b>Plateau at this threshold</b></td>
    <td>How far the threshold can move without the coordination number
    changing, in decades of valence and in &aring;ngstr&ouml;ms of
    distance.</td></tr>
</table>
<p>Below that, every contact found at the tabulation threshold, with its
distance, its partial valence, a filled marker if it is above the bond threshold
and an <i>est.</i> flag if its <i>R</i><sub>0</sub> was estimated rather than
fitted. The footer states the <i>R</i><sub>0</sub>, <i>b</i> and source
actually used.</p>
"""),

        ("lonepair", "The bond-valence vector", f"""
<h1>The bond-valence vector, and the lone pair</h1>
<p><i>View &rsaquo; Show bond-valence vector</i> draws a
lobe on each cation pointing away from its bonded ligands.</p>
<h2>What is drawn</h2>
<p>For each drawn atom, FACET forms
<b>V</b>&nbsp;=&nbsp;&Sigma;<i>v</i><sub>i</sub><b>&ucirc;</b><sub>i</sub> over
the contacts above the bond threshold, where <b>&ucirc;</b><sub>i</sub> points
from the cation to the ligand. The lobe is drawn along
&minus;<b>V</b>/|<b>V</b>| — the direction the bonding is <i>not</i> — and its
length is &phi; &times; the site's mean bond length &times; a display scale you
set. So a centrosymmetric site draws nothing at all, because &phi; is zero
there.</p>
<h2>What that is and is not</h2>
<p>It is the direction and relative size of the bond-valence vector sum, which
is a measurement of the coordination geometry. For a cation with an
<i>n</i>s&sup2; configuration this direction is where an ns&sup2; lone pair is
conventionally described as pointing, which is why the feature is asked for —
but the arrow is the vector sum, not an electron density, and its length is a
drawing scale rather than a distance to anything. FACET does not place a
pseudo-ligand at a literature lone-pair distance: that number would come from
elsewhere while looking like a measurement of your file.</p>
<p>The cone overlay is the other half of the same picture: a real cone of the
measured void half-angle about the void axis. Where several equally wide cones
exist, one axis is returned and the degeneracy is stated in the note.</p>
<p {note}>Both follow the bond threshold, so a lobe that changes markedly as you
drag the explorer is telling you that the vector sum depends on contacts near
the cut.</p>
"""),

        ("utilities", "Tools", f"""
<h1>Tools</h1>
<p>Whole-structure tables, each recomputed for the active structure.</p>
<ul>
<li><b>Cell</b> — lattice parameters, the reciprocal cell in the
crystallographic convention (<i>a</i>* = <b>b</b>&times;<b>c</b>/<i>V</i>,
without 2&pi;), density, composition and formula weight.</li>
<li><b>File</b> — where the structure came from, what every health check
measured, and the displacement parameters. See below.</li>
<li><b>Angles</b> — ligand&ndash;centre&ndash;ligand angles for the selected
site, over the bonded set.</li>
<li><b>Shells</b> — neighbours grouped into shells with their degeneracies,
which is the form a radial analysis wants.</li>
<li><b>Reflections</b> — <i>d</i>-spacings and Bragg angles for a chosen
wavelength. Geometry only; intensities live in the Diffraction tab.</li>
<li><b>CN methods</b> — the same site counted by every rule FACET can apply:
valence-thresholded, ECoN, maximum gap, Voronoi, and optionally eight
pymatgen strategies. They are listed side by side, and none is marked as the
answer.</li>
<li><b>Connectivity</b> — which polyhedra share one ligand, two, or three and
more: corner, edge and face sharing.</li>
<li><b>Cutoffs</b> — the distance each element pair gets from the current
thresholds, with its <i>R</i><sub>0</sub>, <i>b</i>, and where they came
from.</li>
<li><b>Stability</b> — the plateau for every site at once: from where to where
the threshold can move before that site's count changes.</li>
<li><b>Anion valences</b> — bond-valence sums computed <i>around each anion</i>,
by a search in the opposite direction from the cation one. Cation and anion
totals count the same bonds from opposite ends, so their agreement is a check on
the arithmetic rather than a restatement of it. The charge balance line reports
the residual.</li>
</ul>
<p {note}>A tab is computed when you show it, not before. Recomputing all ten on
every move of the threshold took nearly four seconds on a structure of a few
hundred atoms, and nine of them were behind another tab at the time.</p>

<h2>The File tab</h2>
<p>What the file says about itself, and what can be measured about it. Three
parts:</p>
<p><b>Where this came from</b> — the path, the name, the formula as written, the
space group, the database code and the reference. Every number on every other
tab is derived from this file, and the file is not always the one you think it
is.</p>
<p><b>Checks</b> — coordinates that collide, occupancies over one, uncertainties
large enough to swallow a bond length, a formula that disagrees with the atom
list, a bond-valence sum far from the stated oxidation state, a displacement
tensor that describes no ellipsoid. Each says what it measured and the value it
was compared against, grouped by how far outside the ordinary range it fell.
<b>None of them says whether the structure is usable.</b> That depends on what
is being asked of it: coordinates too uncertain to settle a bond-valence sum
still carry the cell that indexes a powder pattern.</p>
<p><b>Displacement</b> — where the file gives anisotropic parameters, the tensor
converted to Cartesian axes and diagonalised. <i>U</i><sub>eq</sub> is one third
of the trace, which is the quantity comparable with a quoted
<i>U</i><sub>iso</sub>; the three r.m.s. values are the displacements along the
principal axes. That last part is what no single number carries — a site can
have an entirely ordinary <i>U</i><sub>eq</sub> and still be four times longer
than it is wide.</p>
<p {note}>A displacement tensor is a covariance matrix and has to be positive
definite. One that is not describes a hyperboloid, which is not a shape an atom
can have, and the site was refined where the data did not constrain it. Such a
site is listed with its eigenvalues rather than with an r.m.s. it does not have.
The test is made on the file's own components, so it does not depend on the
conversion: changing basis cannot change the sign of an eigenvalue.</p>
"""),

        ("diffraction", "Diffraction and PDF", f"""
<h1>Diffraction and the pair distribution function</h1>
<h2>Powder pattern</h2>
<p>The <b>Diffraction</b> tab computes |<i>F</i>|&sup2; from the structure for
X-rays, neutrons or electrons, applies the Lorentz&ndash;polarisation factor and
multiplicity, and convolves with a pseudo-Voigt whose width follows the Caglioti
form FWHM&sup2; = <i>U</i>&thinsp;tan&sup2;&theta; +
<i>V</i>&thinsp;tan&theta; + <i>W</i>. A measured pattern can be loaded and
scaled to the calculation by least squares, with the difference curve
underneath.</p>
<p>Exact from the structure: the reflection positions, the multiplicities, the
structure factors given the tabulated form factors. Assumed: the wavelength, the
peak shape and widths, and the isotropic displacement parameter used where the
file gives none. The notes under the plot say which of those applied.</p>
<h2>Pair distribution function</h2>
<p>The <b>PDF</b> tab computes <i>G</i>(<i>r</i>) from the same structure:</p>
<p align="center"><i>R</i>(<i>r</i>) = (1/<i>N</i>)
&Sigma;<sub><i>i</i>&ne;<i>j</i></sub> <i>w<sub>ij</sub></i>
&thinsp;&Nu;(<i>r</i>; <i>r<sub>ij</sub></i>, &sigma;<sub><i>ij</i></sub>)
&nbsp;,&nbsp;&nbsp; <i>G</i>(<i>r</i>) = <i>R</i>(<i>r</i>)/<i>r</i> &minus;
4&pi;<i>r</i>&rho;<sub>0</sub></p>
<p>with <i>w<sub>ij</sub></i> the scattering weights for the radiation chosen.
The area under a peak of <i>R</i>(<i>r</i>) is the scattering-weighted
coordination number for that shell, which is the quantity a PDF is usually read
for — so it is printed for the peak you select.</p>
<p>Two instrument effects are offered, and both are off until you set them.
Truncating the transform at a finite <i>Q</i><sub>max</sub> is a convolution in
<i>r</i>, and it is what produces the ripple below the first peak; FACET applies
it over the odd extension <i>G</i>(&minus;<i>r</i>) = &minus;<i>G</i>(<i>r</i>),
so that ripple appears where a measurement would show it. Finite
<i>Q</i>-resolution is the dual operation, a multiplication by
exp(&minus;(<i>r</i><i>Q</i><sub>damp</sub>)&sup2;/2).</p>
<p>The peak widths come from the displacement parameters in the file:
&sigma;<sub><i>ij</i></sub>&sup2; = <i>U<sub>i</sub></i> + <i>U<sub>j</sub></i>
for isotropic <i>U</i>. That is the <i>uncorrelated</i> width. Near neighbours
move together, which makes their peaks narrower than this, and the correlation
terms (&delta;<sub>1</sub>, &delta;<sub>2</sub>) are user inputs defaulting to
zero — FACET cannot measure them from one structure.</p>
<p {note}>This is the PDF of the average crystal. A fractionally occupied site
contributes fractional pair correlations, and a real disordered material's
measured PDF differs from it by design — that difference is usually the reason a
PDF was measured.</p>
"""),

        ("exafs", "EXAFS shells and FEFF", f"""
<h1>EXAFS shells and FEFF</h1>
<p>The <b>EXAFS</b> tab turns the structure into the things an EXAFS fit needs,
and reads back what a real scattering calculation produced.</p>
<h2>The shell table</h2>
<p>Per shell about the absorber: element, degeneracy <i>N</i> (and the
occupancy-weighted <i>N</i>), mean <i>R</i> with the spread inside the shell,
which crystallographic sites contribute, and two independently sourced estimates
of &sigma;&sup2; — the uncorrelated value from the file's own displacement
parameters, and an Einstein-model value for a correlated pair at a temperature
and Einstein frequency you set. They are different quantities and are labelled
as such.</p>
<h2>The resolution report</h2>
<p>Over the <i>k</i>-range you intend to fit, the resolution is
&Delta;<i>R</i> = &pi;/(2&Delta;<i>k</i>), and the number of independent points
is <i>N</i><sub>idp</sub> = 2&Delta;<i>k</i>&Delta;<i>R</i>/&pi;. FACET states
which of your shells are closer together than &Delta;<i>R</i>, and how many
parameters your shell list would need against <i>N</i><sub>idp</sub>. Both are
statements about the structure and the range you chose.</p>
<h2>FEFF</h2>
<p><i>File &rsaquo; Export &rsaquo; FEFF input for this site</i> writes a
<code>feff.inp</code> for the selected absorber. FACET does not run FEFF and
does not bundle it; if you have a <code>feff8l</code> or <code>feff6l</code>
executable, point the panel at it and the panel will run it and read the results
back. <code>files.dat</code> and the <code>feffNNNN.dat</code> path files are
parsed to show FEFF's own degeneracy, effective path length, path type and
amplitude ratio beside FACET's crystallographic shells — which is how a
single-scattering count and a real multiple-scattering calculation can be
compared without either being asserted from the other.</p>
<h2>What is declined, and why</h2>
<p>FACET does not compute &chi;(<i>k</i>) from the structure alone. A
single-scattering &chi;(<i>k</i>) without phase shifts puts its Fourier
transform peaks about 0.4 &Aring; away from the distances that generated it, so
a curve produced that way misstates the one quantity it would be read for. Where
FEFF path files are present, &chi;(<i>k</i>) <i>is</i> built — from FEFF's
amplitudes and phases with FACET's degeneracies and distances — and it is
labelled with where each half came from. XANES needs full multiple scattering or
a DFT calculation; FACET writes the input for one and does not pretend to
substitute for it.</p>
"""),

        ("planes", "Planes, slabs and volumes", f"""
<h1>Planes, slabs and volumes</h1>
<h2>Planes</h2>
<p>Enter (<i>hkl</i>) or pick from the presets to draw the lattice plane and
report its <i>d</i>-spacing, the atoms lying on it within a tolerance you set,
and its occupancy. Interplanar angles between two sets of indices are computed
from the reciprocal metric, so they are right in a triclinic cell as well as a
cubic one.</p>
<h2>Slabs</h2>
<p>A slab keeps only what lies between two parallel planes. Bonds are kept only
if both of their drawn endpoints survive, so a slab never leaves a bond hanging
into empty space.</p>
<h2>Volume</h2>
<p>The <b>Volume</b> tab evaluates the bond-valence sum a probe ion of your
choice would have on a grid through the cell, and draws either a section through
it — with contours and a colour map — or an isosurface in the 3D view. The
section is where a migration path or an interstitial site shows itself as a
region of near-correct valence for the probe.</p>
<p {note}>Grid work is bounded: a request large enough to take minutes is
refused with the number it would have needed, rather than freezing the
window.</p>
"""),

        ("disorder", "Disorder and overrides", """
<h1>Disorder and overrides</h1>
<h2>Disorder</h2>
<p>Where a CIF declares assemblies and groups, the <b>Disorder</b> tab lists
them and lets one configuration be shown at a time. Where it declares only
partial occupancies, sites closer together than a physical bond are reported as
mutually exclusive candidates. Which configuration is real is not decided
here.</p>
<h2>Styles</h2>
<p>Any element, site or single atom can be given its own colour, radius or
visibility. The <b>Styles</b> tab lists every one of them so that a figure
can be retraced, and any of them can be removed. Overrides belong to the
structure they were made on, and an atom index that no longer addresses an atom
is dropped when a file is reloaded rather than applied to whatever landed on
that index.</p>
<p>Everything on both tabs is undoable: <i>Edit &rsaquo; Undo</i>, or
Ctrl+Z.</p>
"""),

        ("appearance", "Themes and figures", """
<h1>Themes and figures</h1>
<p><i>View &rsaquo; Theme</i> switches the whole application — viewport, menus,
panels and tables. The presets are a white VESTA-like default, a light theme for
print, a greyscale publication theme with depth cueing off, a high-contrast
theme chosen to survive the commonest colour-vision deficiencies, and two darker
grounds for working at night.</p>
<p>The <b>Theme</b> tab goes further: per-element colours, what an atom's
colour <i>means</i> (element, site, bond-valence sum, coordination number,
&phi;, or valence discrepancy), atom and bond scaling, depth cueing, ambient
occlusion and outlines. A theme can be saved to a file and shared, which is how
a set of figures from several people ends up matching.</p>
<h2>Exporting a figure</h2>
<p><i>File &rsaquo; Export image</i> writes a raster image at a chosen
supersampling. <i>File &rsaquo; Export as vector</i> writes SVG or PDF — real
vector geometry, not a bitmap in a wrapper, so it can be edited in a drawing
program and scales without pixels. The vector exporter is the same code path as
the software renderer, so what it writes is what the fallback tier draws.</p>
<h2>Anion sites</h2>
<p>By default the site list holds the cations, and the anions appear only as
their ligands. <b>Anion sites</b>, under the list, analyses each anion as a site
in its own right: its coordination number, its contacts, its plateau and its own
bond-valence sum, counted from the cations around it.</p>
<p>It is the other half of the same check, and it catches things the cation sums
do not. An oxygen whose sum comes to 1.4 is either missing a bond to something
the file leaves out — a hydrogen, usually — or is not where the refinement put
it. Cation and anion totals count the same bonds from opposite ends, so their
agreement is a check on the arithmetic rather than a restatement of it.</p>
<p {note}>The search is done around the anion rather than gathered from the
cation results. A cation site's contact list belongs to one representative atom
of that site, and an anion is reached by cations from every equivalent position
— most of which are not that representative. Summing the cation lists gives each
anion only the fraction of its bonds that happen to touch a representative,
which on α-Bi₂O₃ came to 0.66 v.u. for an O²⁻: low enough to look like a finding
rather than an error.</p>

<h2>What the view turns about</h2>
<p>By default the view rotates about the middle of the drawn cell block. Right-
click an atom and choose <i>Rotate about &lt;label&gt;</i> to turn about that
atom instead — useful when the site you are examining is at the edge of the
cell and keeps swinging out of frame. The picture does not jump when you choose
it: the camera pans so that the atom is where the middle of the view already
was, and nothing grows or shrinks.</p>
<p>A site you cannot find in the view can be reached from the site list on the
left: double-click it, or right-click it for the same two choices. Where a site
is drawn several times over — once per cell of the block — the copy nearest the
middle of the block is the one used, since turning about one at the edge puts
the rest of the structure off to one side.</p>
<p>The toolbar says which atom it is for as long as it is set. <b>C</b> puts the
centre back at the middle of the cell, <b>Esc</b> clears it along with the
selection and any measurement, and panning with the right or middle button moves
the centre off the atom as a pan always has.</p>
<p {note}>The centre is kept as a position, not as an atom. Nothing that
redraws the structure — moving the threshold, changing the style, cutting a
slab — disturbs it, but if the same atom moves, as it can between disorder
configurations, the centre stays where it was rather than following.</p>

<h2>Displacement ellipsoids</h2>
<p>The <b>Style</b> list has an entry for them. Each atom is drawn as the
surface enclosing a chosen fraction of its displacement distribution — 50% by
default, which is the crystallographic convention and what every other program
draws unless told otherwise. The fraction is chosen beside the style, and it is
shown there rather than hidden in a menu because it is part of what the picture
says: a 50% and a 99% ellipsoid of the same atom differ by more than a factor of
two.</p>
<p>The lines across each ellipsoid are the boundaries of its octants, drawn
where its three principal planes cut the surface. They are what makes it read as
a solid with an orientation rather than as a flat oval, and they are the
convention ORTEP set. An atom drawn without them is one the file gave no
anisotropic tensor for: it is a sphere from <i>U</i><sub>iso</sub>, or, where
there was nothing at all, a small fixed sphere. Those have no principal axes,
and marking axes on them would assert a direction the file never gave.</p>
<p {note}>Unlike every other style, the drawn size here is a measurement rather
than a convention. It is not scaled by occupancy and not scaled by the theme's
atom size, both of which would make it a picture of something else. Bonds are
drawn thin in this style, because a 50% ellipsoid is about a tenth of an
angstrom across and an ordinary bond is drawn wider than that.</p>
<p>A tensor that is not positive definite describes no ellipsoid at all. Such a
site falls back to its isotropic parameter and is listed on the <b>File</b>
tab under the checks, with the eigenvalue that makes it impossible.</p>

<h2>Lighting</h2>
<p>The light is fixed to the viewer, not to the crystal: turning the structure
does not turn the lamp with it, so looking along &minus;c is lit exactly as
looking along +c. Depth cueing, which fades distant atoms towards the
background, is the separate control in the Theme tab, and it is what the
publication theme switches off.</p>

<h2>Stereo</h2>
<p>Anaglyph, greyscale anaglyph, side-by-side and cross-eyed pairs are in the
Theme tab. The separation is an angle rather than a distance, so the depth
survives zooming.</p>
"""),

        ("shortcuts", "Keyboard and mouse", """
<h1>Keyboard and mouse</h1>
<table cellpadding="4" width="100%">
<tr><td width="40%"><b>Left-drag</b></td><td>Rotate</td></tr>
<tr><td><b>Middle-drag</b> or <b>right-drag</b></td><td>Pan. This moves the
   centre of rotation with it.</td></tr>
<tr><td><b>Wheel</b></td><td>Zoom</td></tr>
<tr><td><b>Left-click an atom</b></td><td>Select that site</td></tr>
<tr><td><b>Right-click</b></td><td>Context menu for the atom under the cursor:
   colour, radius, visibility, its polyhedron, and what to turn the view
   about</td></tr>
<tr><td><b>Click two atoms</b></td><td>Distance</td></tr>
<tr><td><b>Click three</b></td><td>Angle</td></tr>
<tr><td><b>Click four</b></td><td>Torsion</td></tr>
<tr><td><b>R</b></td><td>Reset the view</td></tr>
<tr><td><b>C</b></td><td>Turn about the centre of the cell again</td></tr>
<tr><td><b>Esc</b></td><td>Clear the selection, the measurement and the
   rotation centre</td></tr>
<tr><td><b>L</b></td><td>Cycle the atom labels</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open a structure</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Export an image</td></tr>
<tr><td><b>Ctrl+Z</b> / <b>Ctrl+Y</b></td><td>Undo / redo</td></tr>
<tr><td><b>F1</b></td><td>This manual</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>Quit</td></tr>
</table>
<p>A drag that moves more than a few pixels is a rotation, not a click, so
turning the structure and releasing over an atom does not select it.</p>
<p>Files can be dropped onto the window, one or many.</p>
"""),

        ("limits", "What FACET does not do", f"""
<h1>What FACET does not do</h1>
<p>Stated plainly, because a tool that is quiet about its limits gets used
beyond them.</p>
<ul>
<li><b>It does not refine anything.</b> No least-squares against measured data,
no structure solution. Measured patterns can be overlaid and scaled, not
fitted.</li>
<li><b>It does not judge a structure.</b> No number is labelled good, poor or
trustworthy, and no coordination number is marked as the right one. Where a
quantity depends on a choice, the choice is shown next to it.</li>
<li><b>It uses anisotropic displacement parameters only as isotropic
equivalents</b> in the derived quantities, so PDF peak widths and
&sigma;&sup2; estimates use <i>U</i><sub>iso</sub>. The tensor itself is read
and reported on the File tab, and is not yet drawn as an ellipsoid.</li>
<li><b>It does not compute XANES</b>, and it does not compute &chi;(<i>k</i>)
from geometry alone. See the EXAFS section for what it does instead.</li>
<li><b>It ships one bond-valence table, not a compilation.</b> Brese &amp;
O'Keeffe's Table 2 — 108 cations against O, F and Cl — with a few pairs to the
heavier anions from their Table 3. Larger accumulated sets, and the newer
refits, are read from files you supply. A pair outside the table falls back to
the O'Keeffe–Brese estimator and is labelled as estimated wherever it is used:
the estimator differs from a fitted value by 0.05 Å on average and by as much as
0.21 Å, which is a factor of 0.57 to 1.48 on every bond valence.</li>
<li><b>It does not correct measured data.</b> No absorption, no background
subtraction, no <i>Q</i>-space corrections.</li>
</ul>
<h2>Where the numbers were checked</h2>
<p>The repository carries <code>VERIFICATION.md</code>, which records how each
quantity was checked against something with no code in common with it: an
independent implementation, a closed form, a physical law, an invariance, or the
input file's own statement of itself. Five faults invisible from inside the
program were found that way, including a symmetry expansion that silently lost
three quarters of one structure's oxygen while every internal check still
agreed.</p>
<p {note}>If a number here disagrees with one from elsewhere, the threshold and
the parameter set are the first two things to compare — both are printed in
every export.</p>
"""),

        ("licences", "Licences and notices", f"""
<h1>Licences and notices</h1>
<p>FACET's own source code is under the <b>MIT Licence</b>, which places no
restriction on passing it on.</p>
<p>The <b>built application</b> bundles third-party components, and one of them
constrains how a binary may be distributed.</p>
<h2>Qt, under the LGPL-3.0</h2>
<p>Qt is used under the LGPL-3.0. That is permitted for an application under any
licence, but LGPL-3.0 &sect;4 requires that whoever receives the binary be able
to replace the Qt libraries with their own build and re-run it. FACET is
therefore packaged as a <b>directory</b>, not a single self-extracting
executable, so the Qt DLLs sit beside the program as ordinary replaceable files.
A <code>licenses</code> folder next to the executable carries the LGPL text and
a note on how to do the replacement.</p>
<h2>Everything else</h2>
<table cellpadding="3" width="100%">
<tr><td width="35%">gemmi</td><td>MPL-2.0</td></tr>
<tr><td>spglib</td><td>BSD-3-Clause</td></tr>
<tr><td>NumPy, SciPy, pandas</td><td>BSD-3-Clause</td></tr>
<tr><td>matplotlib</td><td>matplotlib licence (PSF-derived)</td></tr>
<tr><td>pymatgen</td><td>MIT</td></tr>
<tr><td>PyInstaller</td><td>GPL-2.0-or-later, with the bootloader
    exception that permits distributing a bundled application under any
    licence</td></tr>
<tr><td>Mesa llvmpipe (<code>opengl32sw.dll</code>)</td><td>MIT</td></tr>
</table>
<h2>Bond-valence parameters</h2>
<p>The published compilations — the IUCr <code>bvparm</code> distribution,
Gagn&eacute; &amp; Hawthorne's tables, softBV — are <b>not</b> bundled, because
each carries its own terms. FACET reads them from a file you supply and records
which set produced a given number. Cite the set you used.</p>
<p {note}>The full text, with the pre-distribution checklist, is
<code>THIRD_PARTY_NOTICES.md</code> in the repository and beside the
executable.</p>
"""),
    ]


def manual_html(theme=None) -> str:
    """The whole manual as one document, for printing or searching."""
    return "".join(body for _key, _title, body in _sections(theme))


# ---------------------------------------------------------------------------
# dialogs
# ---------------------------------------------------------------------------

class HelpDialog(QDialog):
    """One HTML document in a browser, with a Close button.

    The same shape as PRISM's dialog of the same name, so that the two
    applications behave alike for someone moving between them.
    """

    def __init__(self, parent=None, *, html: str = "",
                 title: str = "Help", width: int = 720, height: int = 620):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)


class AboutDialog(HelpDialog):
    """About, with the application mark above the text."""

    def __init__(self, parent=None, *, renderer: str = "", theme=None):
        super().__init__(parent, html=about_html(renderer, theme),
                         title=f"About {NAME}", width=620, height=640)
        from .branding import logo_pixmap

        mark = QLabel()
        mark.setPixmap(logo_pixmap(64, ground=False))
        mark.setAlignment(Qt.AlignCenter)
        self.layout().insertWidget(0, mark)


class ManualDialog(QDialog):
    """The manual: sections on the left, the chosen one on the right.

    A single scrolling document would be simpler, but the manual is long enough
    that finding the paragraph about thresholds in it becomes the hard part.
    """

    def __init__(self, parent=None, *, section: str = "", theme=None):
        super().__init__(parent)
        self.setWindowTitle(f"{NAME} manual")
        self.resize(940, 700)
        self._sections = _sections(theme)

        self.contents = QListWidget()
        self.contents.setMinimumWidth(190)
        for _key, title, _body in self._sections:
            self.contents.addItem(title)
        self.contents.currentRowChanged.connect(self._show_row)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        body = QFont(self.browser.font())
        body.setPointSizeF(body.pointSizeF() + 0.5)
        self.browser.setFont(body)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.contents)
        right = QWidget()
        col = QVBoxLayout(right)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.browser, 1)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([210, 730])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1)
        row = QHBoxLayout()
        self.whole = QPushButton("Show the whole manual")
        self.whole.clicked.connect(self._show_everything)
        row.addWidget(self.whole)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)

        self.show_section(section)

    def _show_row(self, row: int) -> None:
        if 0 <= row < len(self._sections):
            self.browser.setHtml(self._sections[row][2])
            self.browser.verticalScrollBar().setValue(0)

    def _show_everything(self) -> None:
        self.contents.setCurrentRow(-1)
        self.browser.setHtml("".join(b for _k, _t, b in self._sections))
        self.browser.verticalScrollBar().setValue(0)

    def show_section(self, key: str = "") -> None:
        """Open the section named ``key``, or the first one."""
        keys = [k for k, _t, _b in self._sections]
        row = keys.index(key) if key in keys else 0
        self.contents.setCurrentRow(row)
