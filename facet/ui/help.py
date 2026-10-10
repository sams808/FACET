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


def _examples_rows() -> str:
    """The bundled-structure table, built from the catalogue itself.

    Written out by hand it would describe whichever set of files was bundled
    on the day it was written, which is exactly the kind of documentation that
    goes quietly wrong.
    """
    from ..core import examples

    rows = []
    for e in examples.CATALOGUE:
        rows.append(
            f"<tr><td width='24%'><b>{e.title}</b><br>"
            f"<span style='font-size:11px'>{e.formula} &middot; COD {e.cod}</span>"
            f"</td><td>{e.detail}</td></tr>")
    return "\n".join(rows)


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

<p>Written by <b>Sami Soudani</b>, in the <b>McCloy</b> group at
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

<h2>Worked examples</h2>
<p>Six structures are bundled with FACET, under <i>File &rsaquo; Examples</i>.
The six tutorials that follow use nothing else, so they can be worked through
on a new installation with no files to find and no network. Each one states the
numbers you should see, so you can tell whether you are reading the application
correctly.</p>
"""),

        ("tut-quartz", "Tutorial 1 · A number you can trust", f"""
<h1>Tutorial 1 &middot; A coordination number you can trust</h1>
<p {note}>Quartz, SiO<sub>2</sub>. Five minutes. Teaches the workspace, the
Site tab and the cutoff explorer.</p>

<ol>
<li>Open <i>File &rsaquo; Examples &rsaquo; Quartz (SiO<sub>2</sub>)</i>. The
structure appears in the 3D view and the <b>Structures</b> dock on the left
gains a row.</li>
<li>In the <b>Sites</b> dock below it, click <b>Si</b>. The <b>Site</b> tab on
the right now describes that site, and the silicon's polyhedron is drawn.</li>
<li>Read the top of the Site tab. You should see a <b>coordination number of
4</b> and a <b>bond-valence sum near 4.18</b> against a nominal +4.</li>
<li>Look at the contact list lower down: four oxygens, two at 1.605 &Aring; and
two at 1.611 &Aring;, each carrying about 1.0 v.u., then a long gap before
anything else.</li>
<li>Now the point of the application. Drag the vertical line in the <b>cutoff
explorer</b> along the bottom of the window, left and right. Watch the
coordination number in the Site tab.</li>
</ol>

<p><b>What you should see.</b> It stays at 4. The staircase for this site has
one very long tread: the answer survives from 0.020 v.u. up to 1.035 v.u., a
range of <b>1.71 decades</b> &mdash; a more than fiftyfold change in where the
line is drawn. The Site tab reports that as the plateau width.</p>

<p>That is what it looks like when a coordination number is a property of the
structure. Nobody arguing about quartz is arguing about whether silicon is
four-coordinate, and the staircase shows why: there is no choice of threshold,
within reason, that would give any other answer.</p>

<p {note}>The sum of 4.18 rather than exactly 4.00 is the normal spread of
tabulated <i>R</i><sub>0</sub> values, not a fault in the structure. The Site
tab shows the propagated uncertainty beside it.</p>
"""),

        ("tut-eulytite", "Tutorial 2 · And one you cannot", f"""
<h1>Tutorial 2 &middot; And one you cannot</h1>
<p {note}>Eulytite, Bi<sub>4</sub>(SiO<sub>4</sub>)<sub>3</sub>. Ten minutes.
Teaches the contact list, the plateau, and the bond-valence vector. Tutorial 1
first.</p>

<p>The useful thing about this structure is that both kinds of site are in it,
so nothing can be blamed on the data.</p>

<ol>
<li>Open <i>File &rsaquo; Examples &rsaquo; Eulytite</i>.</li>
<li>Select <b>Si</b> in the Sites dock. Coordination number 4, four oxygens all
at 1.623 &Aring;, plateau <b>1.70 decades</b>. This is quartz again: a settled
answer.</li>
<li>Now select <b>Bi</b> in the same structure. Coordination number <b>6</b>,
bond-valence sum 3.24 against +3.</li>
<li>Read the contact list. Three oxygens at <b>2.158 &Aring;</b> and three at
<b>2.606 &Aring;</b>. There is no third group, and nothing obvious separating
the second group from the first.</li>
<li>Drag the cutoff explorer to the right, past 0.25 v.u. The coordination
number drops from 6 to <b>3</b>, and three bonds disappear from the 3D
view.</li>
<li>Turn on <i>View &rsaquo; Show bond-valence vector</i>. A lobe appears on
the bismuth, pointing away from where the bonding is concentrated.</li>
</ol>

<p><b>What you should see.</b> The Bi plateau is 1.09 decades wide, running
from 0.020 to 0.248 v.u. Above 0.248 the answer is 3; below it, 6. Both are
defensible, and the published literature contains both. &phi; for this site is
<b>0.419</b>, against 0.000 for the silicon in the same crystal: the bismuth
environment is strongly one-sided, which is the 6s<sup>2</sup> lone pair.</p>

<p>Compare the two sites' entries in the cutoff explorer. The silicon's
staircase has one long tread; the bismuth's has several short ones. FACET will
not choose between them for you, and that refusal is the point: reporting
"six-coordinate" without saying at what threshold leaves out the only part of
the statement that was a decision.</p>
"""),

        ("tut-polymorphs", "Tutorial 3 · One compound, two answers", f"""
<h1>Tutorial 3 &middot; One compound, two answers</h1>
<p {note}>Senarmontite and valentinite, both Sb<sub>2</sub>O<sub>3</sub>. Ten
minutes. Teaches loading several structures and exporting a table.</p>

<ol>
<li>Open <i>File &rsaquo; Examples &rsaquo; Senarmontite</i>, then
<i>File &rsaquo; Examples &rsaquo; Valentinite</i>. Both now sit in the
<b>Structures</b> dock; click a row to make one active.</li>
<li>With senarmontite active, select the <b>Sb</b> site. Coordination number
<b>6</b>: three oxygens at 1.978 &Aring; and three at 2.902 &Aring;. The
plateau is <b>0.61 decades</b>, which is narrow.</li>
<li>Switch to valentinite and select its <b>Sb</b>. Coordination number
<b>5</b>, at 1.977, 2.019, 2.022, 2.519 and 2.619 &Aring;, with a plateau of
<b>0.94 decades</b>.</li>
<li>Open the <b>Tools</b> tab on the right and look at <b>Cutoffs</b> and
<b>Stability</b>, which tabulate this for every site at once.</li>
<li><i>File &rsaquo; Export &rsaquo; Sites (CSV)</i> writes the active
structure's table; repeat for the other and compare them in a spreadsheet.</li>
</ol>

<p><b>What you should see.</b> The same compound, determined by different
people from different crystals, gives 6 in one polymorph and 5 in the other at
the same threshold. Neither is an error. The bond-valence sums agree closely
&mdash; 3.20 and 3.15 &mdash; because antimony collects the same total valence
either way; what differs is how many neighbours it spreads it over.</p>

<p>This is the comparison to keep in mind when reading a coordination number
out of a paper: it is a statement about one refinement under one rule, and the
rule is usually not stated.</p>
"""),

        ("tut-cryolite", "Tutorial 4 · A number not safe to quote", f"""
<h1>Tutorial 4 &middot; A number not safe to quote</h1>
<p {note}>Cryolite, Na<sub>3</sub>AlF<sub>6</sub>. Ten minutes. Teaches
non-oxide work, the Stability table and the anion view.</p>

<p>Nothing in the method is specific to oxygen. This matters, because the 3.0
&Aring; cutoff that floats around the oxide literature has no standing in a
fluoride, and the bond criterion used here reaches a different distance for
every pair.</p>

<ol>
<li>Open <i>File &rsaquo; Examples &rsaquo; Cryolite</i>.</li>
<li>Select <b>Al</b>. Coordination number 6, sum 2.95 against +3, plateau
<b>1.38 decades</b>. An octahedron nobody would dispute.</li>
<li>Select <b>Na1</b>. Coordination number 6, plateau 0.99 decades. Still
comfortable.</li>
<li>Select <b>Na2</b>. Coordination number 6 &mdash; and a plateau of
<b>0.10 decades</b>, running only from 0.0625 to 0.0779 v.u.</li>
<li>Drag the cutoff explorer slowly across that site. At 0.0625 the answer is
<b>7</b>; by 0.080 it is <b>5</b>; by 0.090 it is <b>4</b>.</li>
<li>Open <b>Tools &rsaquo; Stability</b> to see all three sites ranked by how
much threshold their answer survives, and <b>Tools &rsaquo; Anions</b> to look
at the same structure from the fluorines instead.</li>
</ol>

<p><b>What you should see.</b> A factor of 1.4 in the threshold takes Na2 from
seven-coordinate to four-coordinate. There is no plateau to stand on, so there
is no coordination number to quote &mdash; only a coordination number <i>at a
stated threshold</i>, which is a different and much weaker claim.</p>

<p>Al and Na1 in the same file have wide plateaus. The width is a property of
each site, not of the structure as a whole, and the Stability table is where to
look before quoting any of them.</p>
"""),

        ("tut-hosts", "Tutorial 5 · One ion, two hosts", f"""
<h1>Tutorial 5 &middot; One ion, two hosts</h1>
<p {note}>Eulytite and BiPO<sub>4</sub>. Ten minutes. Teaches comparison across
files and the <i>a priori</i> valences. Tutorial 2 first.</p>

<p>How much of what you measured belongs to the ion, and how much to the
compound it is sitting in? Two files with the same Bi(III) in them answer
it.</p>

<ol>
<li>Open <i>File &rsaquo; Examples &rsaquo; Eulytite</i> and
<i>File &rsaquo; Examples &rsaquo; Bismuth phosphate</i>.</li>
<li>In eulytite, select <b>Bi</b>: coordination number 6, &phi; <b>0.419</b>,
bonds from 2.158 to 2.606 &Aring;.</li>
<li>Switch to BiPO<sub>4</sub> and select <b>Bi</b>: coordination number
<b>8</b>, &phi; <b>0.128</b>, bonds from 2.311 to 2.753 &Aring;. The sum is
much the same, 3.05 against 3.24.</li>
<li>Turn on <i>View &rsaquo; Show bond-valence vector</i> and switch between
the two structures. The lobe is long in eulytite and short in the
phosphate.</li>
<li>Open <b>Tools &rsaquo; A priori</b> in each. This solves the network
equations from the connectivity alone, before any bond length is used, and
reports how much of the asymmetry the topology alone requires.</li>
</ol>

<p><b>What you should see.</b> The same ion, at the same oxidation state,
collecting the same total valence, but over six neighbours in one host and
eight in the other, and far more one-sidedly in the silicate. Phosphorus
competes harder for the oxygens than silicon does, and leaves bismuth a more
even, more distant shell.</p>

<p>So "the coordination number of Bi(III)" is not a property of bismuth. It is
a property of bismuth in a named compound, under a stated rule.</p>
"""),

        ("tut-figure", "Tutorial 6 · Making a figure", """
<h1>Tutorial 6 &middot; Making a figure for a paper or a talk</h1>
<p>Ten minutes. Teaches themes, orientation and vector export.</p>

<ol>
<li>Load any structure and select a site worth showing.</li>
<li>Choose a theme from <i>View &rsaquo; Theme</i>. <b>Light (for print)</b>
and <b>Publication (greyscale)</b> are built for paper; <b>VESTA-like
(white)</b> will look familiar to most readers. <i>More colour
settings&hellip;</i> opens the Theme tab, where individual elements can be
recoloured and the result saved and shared with a group.</li>
<li>Orient the structure. <i>View &rsaquo; Along a</i>, <b>b</b> or <b>c</b>
gives an exact axis view, which is reproducible in a way that dragging is
not.</li>
<li>Set the drawing style and polyhedra from the two rows of controls above the
3D view, and choose how many cells to tile.</li>
<li><i>File &rsaquo; Export vector&hellip;</i> writes <b>SVG</b> or <b>PDF</b>
with real geometry &mdash; lines and polygons that stay sharp at any
magnification and can be edited in Illustrator or Inkscape. Use <i>Export
image&hellip;</i> only when a bitmap is specifically wanted.</li>
</ol>

<p><b>A note on the vector export.</b> It is produced by the software renderer,
which draws the same scene with the same geometry as the hardware one. A figure
exported on a machine with no graphics card is identical to one exported on a
workstation, which is the reason that renderer exists.</p>

<p>For a table rather than a picture, <i>File &rsaquo; Export</i> also writes
the sites and contacts as CSV or XLSX, the cutoff table, and a threshold scan
&mdash; the last being the data behind the staircase, if you would rather plot
it yourself.</p>
"""),

        ("examples", "The bundled structures", f"""
<h1>The bundled structures</h1>
<p>Six structures ship inside the application, under <i>File &rsaquo;
Examples</i>. They are there so the tutorials can be followed on a new
installation with nothing downloaded, and they are chosen to disagree with one
another.</p>
<table cellpadding="4" width="100%">
{_examples_rows()}
</table>
<p>All six come from the Crystallography Open Database, whose contents are in
the public domain, and are redistributed unchanged. The original determination
for each is credited in <code>ATTRIBUTION.md</code> beside the files; cite the
paper rather than the database. Nothing stops you passing FACET on with them
included.</p>
<p {note}><i>Open all of them</i> at the foot of the menu loads the whole set
at once, after which the <b>Structures</b> dock switches between them.</p>
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
<h2>An MD model</h2>
<p>When the row selected in the Structures dock is an MD model, the same
five regions show the Model workspace instead: an Elements dock, the frame
stepper, the frame or a figure, the threshold strip, and the Setup, Results,
Highlight and Notes tabs. See <i>MD models and the Model workspace</i>.</p>
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

        (MD_SECTION, "MD models and the Model workspace",
         _md_section_html(note)),

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
<p>Files can be dropped onto the window, one or many. An MD model or
trajectory among them is listed in the Structures dock and opens the Model
workspace (see <i>MD models</i>), where <b>Page Down</b> / <b>Page Up</b>,
<b>Home</b> and <b>End</b> step through the frames and <b>H</b> opens the
Highlight tab.</p>
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
<li><b>It does not run molecular dynamics.</b> It reads the models and
trajectories a simulation wrote and measures them; it integrates no equation
of motion, fits no potential and edits no model. Nor does it simulate a
quadrupolar NMR spectrum: its NMR analysis gives isotropic shifts from a
published structure-shift correlation whose coefficients and reference are
supplied by the user, none shipping with FACET.</li>
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
# the MD models section
# ---------------------------------------------------------------------------
# The key the Model window's Help menu and the crystal window's Help > MD
# models open the manual at: ManualDialog(parent, section=MD_SECTION).
MD_SECTION = "md"


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _md_formats_rows() -> str:
    """One table row per MD format FACET reads, from the reader's own
    registry (``preview.md_filter_entries``), so a format added there is
    listed here without an edit."""
    try:
        from .preview import md_filter_entries

        entries = md_filter_entries()
    except Exception as error:        # a build without the MD reader
        return (f"<tr><td colspan=2>The MD reader could not be loaded "
                f"({_esc(error)}).</td></tr>")
    return "".join(
        f"<tr><td>{_esc(label)}</td><td><code>{_esc(' '.join(patterns))}"
        "</code></td></tr>" for label, patterns, _name in entries)


def _md_analyses_rows() -> str:
    """One row per analysis of ``md_analysis``: what it measures (its own
    summary) and the inputs it needs that have no default, as
    ``missing_inputs`` names them for a request that gives none, so the
    table states what the engine refuses rather than a copy of it."""
    try:
        from ..core import md_analysis as ma
    except Exception as error:        # a build without the MD engine
        return (f"<tr><td colspan=3>The MD analyses could not be loaded "
                f"({_esc(error)}).</td></tr>")
    try:
        # the setup's own labels and wording, so the manual names each input
        # as the Model window labels it
        from .md_dialogs import OPTIONAL_OUTPUTS, field_label, user_text
    except Exception:                 # a build without the Model window
        OPTIONAL_OUTPUTS = {}

        def field_label(name):
            return name

        def user_text(text):
            return text

    timed = set(getattr(ma, "_TIMED", ma.TRAJECTORY_ANALYSES))
    rows = []
    for name in ma.ANALYSES:
        try:
            request = ma.AnalysisRequest(analyses=(name,))
            needs = [m for m in ma.missing_inputs(request)
                     if name in m.analysis.split("/")]
        except Exception as error:    # stated, not hidden
            needs = []
            inputs = f"(not listed: {_esc(error)})"
        else:
            inputs = "<br>".join(
                (f"{_esc(field_label(m.name))} (<code>{_esc(m.name)}</code>)"
                 if m.name not in ("formers", "analyses")
                 else f"<code>{_esc(m.name)}</code>")
                + f": {_esc(user_text(m.why))}" for m in needs)
        if name in timed:
            inputs += ("<br>" if inputs else "") + (
                "a time axis (the file's frame times, or the MD timestep or "
                "frame interval given) and at least two frames")
        if name == "kinetic-temperature":
            inputs += "<br>velocities in the file"
        if name == "vacf":
            inputs += ("<br>velocities in the file, unless they are taken "
                       "from the positions (<code>dynamics.velocities</code> "
                       "'finite difference')")
        optional = [f"{_esc(what)}, given "
                    + " and ".join(f"{_esc(field_label(f))} "
                                   f"(<code>{_esc(f)}</code>)" for f in fields)
                    for what, fields in OPTIONAL_OUTPUTS.get(name, ())]
        if optional:
            inputs += ("<br>" if inputs else "") + "<i>Optional:</i> " \
                + "; ".join(optional)
        rows.append(f"<tr><td><b>{_esc(name)}</b></td>"
                    f"<td>{_esc(ma.ANALYSIS_SUMMARIES[name])}</td>"
                    f"<td>{inputs or 'none beyond the model'}</td></tr>")
    return "".join(rows)


def _md_presets_rows() -> str:
    """One row per preset of the Setup page, from ``md_presets`` itself."""
    try:
        from ..core import md_presets

        presets = md_presets.PRESETS
    except Exception as error:        # a build without the MD engine
        return (f"<tr><td colspan=2>The presets could not be loaded "
                f"({_esc(error)}).</td></tr>")
    return "".join(f"<tr><td width='28%'><b>{_esc(p.name)}</b></td>"
                   f"<td>{_esc(p.description)}</td></tr>" for p in presets)


def _md_questions_rows() -> str:
    """The Setup tree's groups, each with its analyses, from the page."""
    try:
        from .md_setup import ANALYSES_BY_QUESTION

        groups = ANALYSES_BY_QUESTION
    except Exception as error:        # a build without the Model workspace
        return (f"<tr><td colspan=2>The Setup tree could not be loaded "
                f"({_esc(error)}).</td></tr>")
    return "".join(
        f"<tr><td width='28%'><b>{_esc(group)}</b></td><td>"
        + ", ".join(f"{_esc(text)} (<code>{_esc(name)}</code>)"
                    for name, text in members) + "</td></tr>"
        for group, members in groups)


# Where F1 opens the MD section for each tab of the Model workspace (and for
# the read panel): the anchors of _md_section_html.
MD_ANCHORS = {"Setup": "md-setup", "Results": "md-results",
              "Highlight": "md-highlight", "Notes": "md-notes",
              "read-options": "md-open"}


def md_anchor_for(tab: str) -> str:
    """The anchor of the MD section that documents ``tab`` ('' when none)."""
    return MD_ANCHORS.get(str(tab), "")


def _md_section_html(note: str) -> str:
    """The manual's MD models section: what opens and how, the Model
    workspace tab by tab (Setup with its presets and its tree by question,
    Results, Highlight with its rules and channels, Notes), the frames and
    the threshold strip, the Model menu, export, the command line, and what
    is not done. Numbers come from the code (the thresholds, the mass
    tolerance, the ddof of the spread, the atoms-only limit)."""
    try:
        from ..core import md_readers, md_stats

        mass_tol = f"{md_readers.MASS_TOL_AMU:g}"
        ddof = f"{md_stats.STD_DDOF:d}"
    except Exception:                 # a build without the MD engine
        mass_tol, ddof = "the reader's tolerance in", "1"
    try:
        from ..core import md_analysis as ma

        former_readers = ", ".join(ma.FORMER_READERS)
        no_formers = ma.NO_FORMERS
    except Exception:
        former_readers, no_formers = "the network analyses", "none"
    try:
        from .md_workspace import ATOMS_ONLY_ABOVE, CLOSE_WAIT_MS

        atoms_only = f"{ATOMS_ONLY_ABOVE:d}"
        close_wait = f"{CLOSE_WAIT_MS / 1000:g}"
    except Exception:                 # the workspace not in this build
        atoms_only, close_wait = "a few thousand", "a few"
    try:
        from .md_highlight import (DIM_ATOM_RADIUS, DIM_BOND_RADIUS,
                                   DIM_KEEP)

        dim = (f"{100 * DIM_KEEP:.0f} % of its colour over the background, "
               f"{100 * DIM_ATOM_RADIUS:.0f} % of its radius (a bond "
               f"{100 * DIM_BOND_RADIUS:.0f} %)")
    except Exception:
        dim = "a fraction of its colour and radius"
    return f"""
<h1>MD models and the Model workspace</h1>
<p>FACET reads the models and trajectories that molecular-dynamics programs
write and measures them: coordination by bond valence and by distance, the
network (Q<sup><i>n</i></sup>, speciation, rings, connectivity), scattering,
local order, spectroscopy and dynamics. Each descriptor is measured on every
frame chosen and reported as the mean over those frames with its spread,
alongside the provenance that produced it. FACET runs no simulation.</p>
<p>A model is listed in the <b>Structures</b> dock beside the crystals, on
two lines (its name; its atoms and frames). Selecting it switches the window
to the <b>Model workspace</b>: the Sites dock becomes an <b>Elements</b>
dock, the toolbar steps through the frames, the centre shows the frame or a
figure, the strip along the bottom is the frame's threshold staircase, and
the right column holds four tabs, <b>Setup</b>, <b>Results</b>,
<b>Highlight</b> and <b>Notes</b>. Selecting a crystal row brings the crystal
workspace back; both keep their state. <b>F1</b> opens this section at the
part that matches the tab shown.</p>

<a name="md-open"></a>
<h2>What opens as an MD model</h2>
<p>A file is recognised by its content, whatever its name: a LAMMPS dump saved
as <code>.txt</code> opens as a dump, and a CIF opens as a crystal whatever it
is called. The names below are the ones the file dialog lists.</p>
<table cellpadding="3" width="100%">
<tr><td width="34%"><b>Format</b></td><td><b>File names</b></td></tr>
{_md_formats_rows()}
</table>
<p {note}>A plain XYZ trajectory with no <code>Lattice=</code> on its comment
line (LAMMPS's <code>dump xyz</code>, CP2K's <code>pos.xyz</code>) states no
periodic box. It opens all the same, and the box is then taken from another
file of the same run (its LAMMPS data file, CP2K's <code>.cell</code> file)
or typed in; no box is invented.</p>

<h2>Opening one</h2>
<ul>
<li><b>Drop it</b> on the window. It is listed in the Structures dock as a
model and read at once; CIFs dropped with it load as crystals, as before.</li>
<li><i>File &rsaquo; Open MD model&hellip;</i> lists every MD format above.
<i>File &rsaquo; Open&hellip;</i> takes MD files too, under <i>All
files</i>.</li>
<li><i>File &rsaquo; Open MD series as one model&hellip;</i> reads several
files of one run (<code>dump.0.lammpstrj</code>,
<code>dump.1000.lammpstrj</code>, &hellip;) as one trajectory, in the natural
order of their names: digits compare as numbers, so dump.20 comes before
dump.100.</li>
<li>Files dropped together are one model only when their format writes one
snapshot per file (LAMMPS <code>dump cfg</code>) and they share a folder and
a name up to its digits. Any other files are listed one model each, so two
runs named <code>glass_300K</code> and <code>glass_600K</code> are never
averaged into one.</li>
<li>From the command line, <code>py -3.11 -m facet glass.lammpstrj</code>
opens FACET with that model listed and shown.</li>
<li><i>File &rsaquo; Save session</i> records a model as its file, the read
options it was read with and the request of its Setup, never a result; the
session reopens the model and fills Setup again.</li>
</ul>

<h2>Type maps and the read panel</h2>
<p>LAMMPS numbers its atom types, and a number is not an element. An element
is taken only from what a file states: an element column, type labels, or a
mass that matches one element's standard atomic weight to within {mass_tol}
amu (a LAMMPS data file's <code>Masses</code> section names the types of a
dump of the same run). A whole-number mass names no element, since force
fields often round masses to integers. Nothing is guessed: a type that no
source names stays unnamed until an element is given for it. The centre then
shows the <b>read panel</b>: the reader's message in full and, for each type,
the reader's evidence (its label, element column or mass, and its atoms in
the first frame), with a field for the element; the masses can also come
from a LAMMPS data file of the same run, offered with a button when it was
dropped with the dump. A topology (DCD, XTC, AMBER NetCDF) and a box (a
plain XYZ) are asked for the same way, with no element filled in. <i>Read
again</i> reads the file with what was typed; the read options stay under
<i>Setup &rsaquo; More&hellip;</i> afterwards, to read it again with
others.</p>

<a name="md-setup"></a>
<h2>The Setup tab</h2>
<p>The few inputs a run usually needs, in one column, over the complete set
of options (<i>More&hellip;</i> and the fields a greyed analysis names).</p>
<p><b>Preset.</b> A preset is what a user who opens a glass model and asks one
question would tick by hand: the analyses that answer it, the network formers
among the cations present, and the method inputs those analyses require and
have no default for (the ring criterion and largest ring, the void radii, the
free-volume probe and grid, the scattering window and radiations). Nothing
physical is filled in: no timestep, no temperature, no charges, no measured
curve, no absorber. What a preset needs and cannot give is said back as a
note beside the box. Every value stays editable; any edit flips the preset to
<i>Custom</i>. The same ten are under <i>Model &rsaquo; Presets</i>.</p>
<table cellpadding="3" width="100%">
{_md_presets_rows()}
</table>
<p><b>Composition</b> is frame 0 as read. <b>Formers</b> are chips, one per
cation present, none ticked at first: FACET assumes no network former. The
analyses that read formers ({_esc(former_readers)}) stay greyed, with the
reason beside them, until formers are ticked; <i>No former to name</i> under
More&hellip; (<code>{_esc(no_formers)}</code>) runs the glass analysis
without the former-dependent descriptors. <b>Frames</b> are first, last and
every n-th readable frame; the line under them states the frames, their
timesteps and whether the file holds frame times. <b>Threshold</b> is
<i>v</i><sub>bond</sub>; <i>v</i><sub>list</sub>, the parameter file, the
time axis and the oxidation states (model inputs, never resolved from the
geometry) are under More&hellip;. A preset that needs a time axis the file
lacks shows the timestep field in the column.</p>
<p><b>Analyses, by the question they answer.</b> A tick runs an analysis; a
group's box ticks its children. The second column says what a runnable
analysis runs with (its method inputs, and the analyses whose results it
reads) or what a greyed one needs; double-clicking an analysis goes to its
inputs, in a small dialog when the page itself does not show them.</p>
<table cellpadding="3" width="100%">
{_md_questions_rows()}
</table>
<p><b>Run analyses</b> (Ctrl+R) sits under the column with the line that says
what will run, or what the run still needs. While it runs, a progress bar
shows the engine's stage and <b>Cancel</b> stops it after the frame in
progress; what was finished is kept as a cancelled run, and the last
complete run is never replaced by it.</p>

<h2>Oxidation states and formers</h2>
<p>Oxidation states are inputs of the model: each element takes its common
state unless another is given, and an element with no common state needs
one. They decide which atoms are cations and which anions in the bond-valence
split, and which parameters apply. The network analyses (rings, coordination
sequences, polyhedral sharing, components) need named formers, or their graph
elements, centres and T elements given explicitly, and say so beside their
boxes. No timestep, temperature, charge, NMR coefficient or radius is assumed
either.</p>

<a name="md-results"></a>
<h2>The Results tab</h2>
<p>One line names the run (file, frames, thresholds, formers, FACET version);
<i>Provenance</i> unfolds the header every export carries, and <i>Export
all&hellip;</i> writes every descriptor. The tree lists each analysis with
its descriptors (those that differ only by element gathered under one node,
with their mean &plusmn; spread); the filter field hides the names that do
not contain its text. Clicking a descriptor draws its figure and rows in the
centre, with the notes of the analysis under them; the toolbar's
<b>3D view</b> brings the frame back and <b>Figure</b> the descriptor. An
analysis that produced nothing is listed with the reason the engine gave.</p>

<h2>Frames and the spread</h2>
<p>Each descriptor is measured frame by frame and reported as the mean over
the frames used with the sample standard deviation across them
(ddof&nbsp;=&nbsp;{ddof}): the spread between frames, not a standard error,
because the frames of one trajectory are correlated. A frame that cannot be
read is left out and named with its reason, and the provenance of every
result lists the frames it used.</p>

<h2>The analyses</h2>
<p>The per-frame analyses share one neighbour search per pass over the
frames; a second pass runs only when a cutoff comes from the frame-averaged
<i>g</i>(<i>r</i>), and the provenance states the searches each frame
received. The dynamics read the unwrapped positions of the chosen frames
once more. A run that lacks an input is refused before any frame is read,
with every missing input named at once; the inputs in the right-hand column
have no default, and Setup labels each one as written there, with its
request name. Method choices have defaults, each stated in the provenance:
most set a resolution (bin widths, grid steps, the first-minimum rule), and
some change the numbers themselves. Removing the centre-of-mass drift, off by
default, changes every MSD and every diffusion coefficient fitted from it;
an MSD figure states the centre of mass's own MSD when the drift is left
in.</p>
<table cellpadding="3" width="100%">
<tr><td width="18%"><b>Analysis</b></td><td width="37%"><b>What it
measures</b></td><td><b>Inputs it needs</b></td></tr>
{_md_analyses_rows()}
</table>
<p {note}>Checked against the model when the run starts: an element an option
names that the model does not hold, formers none of which are present, a
scattering range beyond half the box, a frame selection that selects
nothing.</p>

<a name="md-frames"></a>
<h2>The frame, the threshold strip and the Elements dock</h2>
<p>The centre draws one frame from that frame's own valence table (the bulk
engine's one search), so the drawn bonds are the contacts the coordination
numbers count, each as thick as its valence, with the ones below
<i>v</i><sub>bond</sub> thin and faded. The toolbar steps through the frames
(<i>View &rsaquo; Frame</i>: Page Down and Page Up, Home and End); the
drawing style, the atom and bond labels, the box and the projection are
beside it, and its note states the frame's timestep and bond count. Without
OpenGL (the QPainter renderer), a frame of more than {atoms_only} atoms is
drawn as atoms only, with a note saying so; the OpenGL renderers draw its
bonds. The status bar names the renderer in use and the time the frame took
to draw.</p>
<p>The <b>threshold strip</b> along the bottom is the mean coordination number
of each element against <i>v</i><sub>bond</sub> on the frame shown, from
<i>v</i><sub>list</sub> ({bv.V_LIST_DEFAULT:g} v.u. unless another is set)
to the frame's largest valence on a logarithmic scale; the vertical line is
the current <i>v</i><sub>bond</sub> ({bv.V_BOND_DEFAULT:g} v.u. unless
another is set), and a dashed one the threshold of the last run when it
differs. Dragging the slider, typing a value or clicking the staircase moves
<i>v</i><sub>bond</sub>: the drawn bonds restyle, the Elements dock and the
highlight rules follow, and no neighbour search is run. One frame is shown,
never an average; the frame-averaged tables of a run keep the
<i>v</i><sub>bond</sub> the run used, and carrying another into every table
is a new run. The distance-cut numbers do not depend on
<i>v</i><sub>bond</sub>.</p>
<p>The <b>Elements</b> dock lists each element of the frame with its atoms,
its oxidation state and its mean CN at the strip's threshold. Clicking an
element draws its atoms in full and the others dimmed; clicking it again
draws every atom alike.</p>

<a name="md-highlight"></a>
<h2>The Highlight tab</h2>
<p>Rules that change the 3D view of the frame shown. Each can be switched off
or removed, and the view follows after a short pause; the rules edit the
drawn arrays, so every renderer tier draws them, the QPainter one included,
and the exported figure carries them. The line under the rules states the
frame, <i>v</i><sub>bond</sub> and what was drawn with which values. A rule
the frame cannot serve (a descriptor that needs formers, a channel backend
this build lacks) is greyed with its reason, never drawn as something
else.</p>
<p><b>Colour atoms by</b> element, bond-valence sum, coordination number or
&phi; (the crystal workspace's colour modes, on the frame at
<i>v</i><sub>bond</sub>, the cations on the ramp and the anions in their
element colour); by Q<sup><i>n</i></sup> or by modifier-rich O, which read
the formers ticked in Setup or those of the last run; or by channel
membership, which needs a channels rule that is on.</p>
<p><b>Show only</b> [an element or any] where [CN, former CN, BVS, &phi; or
Q<sup><i>n</i></sup>] [&le; or &ge;] a value. CN counts bonds to every
counter-ion at <i>v</i><sub>bond</sub>; former CN counts an anion's bonds to
the formers only (an O with former CN 1 is non-bridging); Q<sup><i>n</i></sup>
is a former cation's bonds to anions bonded to two or more formers. With
<i>Dim the rest</i> on, an atom outside the rule keeps {dim}, so the inside
of a box stays visible; off, it is left out of the scene. The count of atoms
matched is written under the rules.</p>
<p><b>Voids</b> at or above a volume in &Aring;&sup3;, drawn as translucent
spheres: the empty spheres of the frame (FACET's van der Waals radii), each
with the radius of a sphere of that volume or more. An elongation above 1
keeps only the spheres whose void region (merged empty spheres) is at least
that many times longer than it is wide; it reads the void regions of
<code>facet.core.md_channels</code> and is greyed with the reason when the
build lacks them.</p>
<p><b>Channels</b>, three ways of measuring where a mobile ion has room, each
reading <code>facet.core.md_channels</code> on the frame shown:</p>
<ul>
<li><b>by charge</b>: the bond-valence landscape of the probe ion (its sum on
a grid of the stated spacing, from the anions within <i>r</i><sub>cut</sub>),
the regions where the mismatch to the probe's valence is at most &Delta;,
and the isosurface at &Delta; drawn in the view; the count line states the
regions, the fraction of the box they fill and which box axes they span.
Measured, not inferred: a connected region is a statement about the sum, not
about a path an ion takes.</li>
<li><b>by modifier density</b>: the anions with <i>k</i> or more atoms of the
modifier element within the M&ndash;anion cutoff, and the clusters they form
(bonded through the formers when they are named).</li>
<li><b>by voids</b>: the void regions of the frame for the probe radius
given, kept when at least as elongated as the rule says.</li>
</ul>
<p>Under channel membership each atom in a region takes the region's colour
and the atoms outside every region are dimmed. The by-charge landscape of
3 000 atoms on a 1 &Aring; grid takes seconds and is computed on a worker
thread; the status bar says so, and the window stays usable.</p>

<a name="md-notes"></a>
<h2>The Notes tab</h2>
<p>Every note and provenance line of the run on show, in order: the run
header (file, frames used and left out, type map and its source, oxidation
states, bond-valence set and thresholds, formers, cutoffs, method
parameters, FACET version), then each analysis's notes, provenance and time,
the timings, the frame shown with its highlight line, and the load summary
(what the reader read and assumed). A refused run states its reason here
with the notes of the run that stays on show.</p>

<a name="md-menu"></a>
<h2>The Model menu</h2>
<p>Every item acts on the model selected in the Structures dock. <i>Run
analyses</i> (Ctrl+R) and <i>Cancel run</i>; <i>Presets</i>, the ten of the
Setup tab; <i>Export results</i> as CSV files or an XLSX workbook, and the
figure or the rows shown; <i>Save request&hellip;</i> writes the setup as a
TOML request file, <i>Load request&hellip;</i> fills the setup from one;
<i>Show the last complete run</i> and <i>Show the cancelled run's partial
result</i> switch between the two runs the model keeps. <i>View &rsaquo;
Frame</i> steps through the frames and <i>View &rsaquo; Highlight&hellip;</i>
(H) opens the rules. Closing the window during a run cancels it and waits at
most {close_wait} s for the engine to notice; work that takes longer finishes
in the background.</p>

<a name="md-export"></a>
<h2>Export</h2>
<p>Results go out as CSV files, one per descriptor with an index, or as one
XLSX workbook (which needs the openpyxl package). Every file starts with the
provenance of its numbers: the model file, the frames used and those left
out, the type map and its source, the oxidation states, the bond-valence
parameter set, <i>v</i><sub>bond</sub> and <i>v</i><sub>list</sub>, the
cutoffs and where they came from, the method parameters and the FACET
version. Numbers are written in full. Figures export as SVG or PDF, which
are vector, and as PNG at 600 dpi; <i>File &rsaquo; Export image</i> and
<i>Export vector</i> write the frame shown with its highlight rules.</p>

<a name="md-cli"></a>
<h2>From the command line</h2>
<p>The same engine runs without a window, on a workstation or a cluster
node:</p>
<pre>py -3.11 -m facet.md describe dump.lammpstrj --type-map 1=Si,2=O,3=Na
py -3.11 -m facet.md analyses
py -3.11 -m facet.md template --out request.toml
py -3.11 -m facet.md analyse dump.lammpstrj --type-map 1=Si,2=O,3=Na \\
    --formers Si --frames 0:100:5 --out results.xlsx
py -3.11 -m facet.md analyse dump.lammpstrj --request request.toml</pre>
<p><code>describe</code> says what the reader finds, <code>analyses</code>
lists the analyses and their inputs, <code>template</code> writes a request
file holding every option, and <code>analyse</code> runs them and writes the
results with their provenance. A request file saved from the Model menu
repeats a run of the window exactly. Without <code>--only</code>, every
analysis whose inputs are given runs and the others are listed with what
they lack.</p>

<h2>What is not done</h2>
<ul>
<li>No molecular dynamics is run, and no model is edited or written back.</li>
<li>No quadrupolar NMR spectrum is simulated. The NMR analysis gives
isotropic shifts and their spectrum from a published structure-shift
correlation, whose coefficients and reference are supplied by the user; none
ships with FACET.</li>
<li>No element, timestep, temperature, charge, network former or NMR
coefficient is assumed.</li>
<li>No result is labelled as a match or a mismatch with experiment: a
comparison with measured data reports the two side by side with their
difference. No channel is labelled a conduction path: the rules draw what a
landscape, a density or a void set measures on one frame.</li>
</ul>
"""


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

    def show_section(self, key: str = "", anchor: str = "") -> None:
        """Open the section named ``key``, or the first one, scrolled to
        ``anchor`` (an ``<a name>`` of that section) when one is given."""
        keys = [k for k, _t, _b in self._sections]
        row = keys.index(key) if key in keys else 0
        if self.contents.currentRow() != row:
            self.contents.setCurrentRow(row)
        else:
            self._show_row(row)
        if anchor:
            self.browser.scrollToAnchor(anchor)
