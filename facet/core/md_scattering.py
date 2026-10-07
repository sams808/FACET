"""Total scattering from an MD model: partial and total S(Q), F(Q), G(r), D(r), T(r).

What a diffraction experiment on a glass measures is one weighted sum of the
partial pair correlations; what an MD model gives is every partial on its own.
This module goes from the second to the first, so that a model can be put next
to a measured S(Q) or G(r), and it does so by two routes that share no code,
so each checks the other.

WHAT IS COMPUTED, AND IN WHOSE CONVENTIONS
------------------------------------------
Every name below is written out, because the same letters mean different
functions in different communities [1, 2, 3].

* **Partial pair distribution functions** g_ab(r), one per unordered element
  pair, from one frame (:func:`frame_partials`), computed by
  ``glass.partial_rdf`` so that the glass descriptors and the scattering read
  one g(r): the pairs of the frame's one pair search (``bulk.iter_pairs``)
  are deposited on the grid of ``pdf.pair_distribution`` (r = dr, 2 dr, ...)
  with ``pdf._deposit``, which splits each distance linearly between its two
  grid points and so conserves the count exactly, and normalised as pdf.py
  normalises: g_ab(r) = H_ab(r) / (N_a 4 pi r^2 rho_b dr), with
  rho_b = N_b / V, so g_ab = g_ba. Where glass.py cuts its grid at half the
  smallest perpendicular box width with a note, :func:`frame_partials`
  refuses an r_max beyond it, so every frame of a trajectory has the same
  grid and no transform runs on a grid shorter than the one asked for.
* **Faber-Ziman partial structure factors** [4] by the sine transform
  (:func:`partial_structure_factors`)::

      S_ab(Q) = 1 + 4 pi rho0 Int_0^r_max r^2 [g_ab(r) - 1] M(r) sin(Qr)/(Qr) dr

  with rho0 = N / V the total number density. The integral is the rectangle
  sum over the grid, the rule ``pdf.forward_transform`` uses, which is the rule
  that keeps a linearly deposited count exact. ``r_window`` sets M(r): 'none'
  (M = 1, a step at r_max) or 'Lorch', M(r) = sin(pi r / r_max)/(pi r / r_max),
  the function Lorch introduced on the Q side [5] used here on the r side.
  There is no default: the choice changes S(Q) and the caller states it.
* **Total structure factors** (:func:`total_structure_factor`), Faber-Ziman
  weighted, for neutrons, X-rays and electrons::

      S(Q) - 1 = sum_a sum_b c_a c_b f_a(Q) f_b(Q) [S_ab(Q) - 1] / <f(Q)>^2

  over ordered pairs, <f> = sum_a c_a f_a, so the weights sum to 1 at every Q.
  F(Q) = Q [S(Q) - 1] (the PDF community's reduced structure function [2]);
  F_K(Q) = <f>^2 [S(Q) - 1] = sum c_a c_b f_a f_b [S_ab - 1] (Keen's F(Q) [1],
  written F_K by Peterson et al. [2, 3]).
* **Real-space functions** (:func:`real_space_from_partials`), in Keen's
  family as Peterson et al. restate it [1, 2, 3]::

      G(r)   = 4 pi r rho0 [g(r) - 1],  g(r) - 1 = sum w_ab [g_ab(r) - 1]
      G_K(r) = sum c_a c_b b_a b_b [g_ab(r) - 1] = <b>^2 G(r) / (4 pi rho0 r)
      D(r)   = <b>^2 G(r)                              ([2] eq. 28)
      T(r)   = <b>^2 [G(r) + 4 pi r rho0] = D(r) + 4 pi r rho0 <b>^2

  G(r) here is ``pdf.pair_distribution``'s G(r), computed the same way (R(r)
  from the deposited, weighted pair counts, G = R/r - 4 pi r rho0), and with
  ``sigma_ang`` set it is broadened by ``pdf._broaden`` as pdf.py broadens. For
  X-rays and electrons the weights are taken at Q = 0, as pdf.py does. A
  measured X-ray G(r) is instead the transform of an S(Q) normalised by
  <f(Q)>^2 at every Q; :func:`real_space_from_sq` computes that form, and
  :func:`analyse_trajectory` returns it beside the Q = 0 form
  (``pdf_g_from_sq``) whenever a Qmax is given. An outside check measured
  the two X-ray forms of a 3 000-atom SiO2 glass up to 0.91 apart at
  Qmax = 25 Å^-1, against a largest |G| of 11.65 (neutron: 0.005);
  re-measured 2026-10-07 on its last frame with r_window 'none' and a
  boxcar Qmax: 0.905 against 11.66 (neutron: 0.005 below r = 15 Å). That
  G(r) is transformed from the sine route's S(Q), so it carries the
  r-window: with 'Lorch' it is M(r) G(r) terminated, G(r) damped towards
  r_max (a test holds this to 3e-3 of max |G|), and on the same frame it
  differed from the Q = 0 form by 0.135-0.21 for neutrons. T(r) is
  written with 4 pi r rho0: Peterson et al. [2] print eq. (29) with
  4 pi r^2 rho0, which does not have the dimensions of G(r) (Å^-2); the text
  of Keen [1] was not available here to compare. A finite Qmax is applied as
  pdf.py applies it: ``pdf.apply_termination`` for a boxcar window (a
  convolution over the odd extension), and for 'Lorch' the sums of
  ``pdf._lorch_round_trip`` with its two sine matrices built once
  (:class:`_LorchOperator`; a test holds the two equal to 1e-12), so a
  trajectory pays for the sines once. D(r)
  is the terminated <b>^2 G(r); T(r) adds the unterminated 4 pi r rho0 <b>^2.
  :func:`real_space_from_sq` is the other direction, G(r) from a total S(Q)
  with ``pdf.inverse_transform``. One property of ``pdf.apply_termination``
  shows near r_max and is pdf.py's, reported and not changed here: its
  ``np.convolve(..., 'same')`` keeps lags up to r_max only, so the image term
  K(r + r') of the odd extension is left out wherever r + r' > r_max. With
  that term written out, the exact discrete sum
  sum_k G_k dr [K(r - r_k) - K(r + r_k)] and the transform route (forward,
  cut at Qmax, back) agreed to 3.8e-5; ``apply_termination`` matched the sum
  without the image lags beyond r_max to 2.3e-15, and differed from the full
  sum by up to 0.0030, 0.37 % of max |G|, at r = 14.0 Å of r_max = 14.9 Å
  (a 1 807-atom disordered model, Qmax = 20 Å^-1, measured 2026-10-06).
* **Bhatia-Thornton** number-concentration structure factors [6] for a binary
  model (:func:`bhatia_thornton`), from the Faber-Ziman partials::

      S_NN = c_A^2 S_AA + c_B^2 S_BB + 2 c_A c_B S_AB
      S_NC = c_A c_B [c_A (S_AA - S_AB) - c_B (S_BB - S_AB)]
      S_CC = c_A c_B [1 + c_A c_B (S_AA + S_BB - 2 S_AB)]

  the forms of Salmon [7], with A the first element in alphabetical order and
  the concentration mode C(q) = c_B rho_A(q) - c_A rho_B(q). A test computes
  S_NN, S_NC and S_CC a second way, from these number and concentration
  modes at every reciprocal-lattice vector, which is Bhatia and Thornton's own
  definition, and finds the same numbers.

THE SECOND ROUTE: S(q) FROM THE POSITIONS
-----------------------------------------
:func:`sq_from_positions` never forms g(r). At every vector q of the box's
reciprocal lattice with 0 < |q| < q_max it sums rho_a(q) = sum_{j in a}
exp(i q.r_j) over the atoms of each element, and the Faber-Ziman partial at
that q is S_ab - 1 = [Re(rho_a rho_b*) / N - c_a delta_ab] / (c_a c_b) (the
relation to the Ashcroft-Langreth partials [8] is eq. 2.35 of Fischer et
al. [9]); the total for a radiation is 1 + (|sum_a f_a rho_a|^2 / N - <f^2>) /
<f>^2, with f at that |q|. Values are averaged over shells of |q|. For the
periodic model this is exact at each lattice vector. It is the intensity
|sum_j b_j exp(i q.r_j)|^2 / N in the Faber-Ziman normalisation, the one the
sine route gives; normalised by <b^2> instead, the same intensity reads
1 + (<b>^2 / <b^2>) [S(q) - 1]. The sum is factorised along the three
reciprocal axes (exp(i 2 pi (h f1 + k f2 + l f3)) is a product of three
factors), so each (h, k) row is one matrix product; q and -q give the same
value, so only h >= 0 is summed and the h > 0 vectors count twice.

Where each route holds. The sine route stops at r_max, at most half the
smallest perpendicular width of the box (:func:`frame_partials` refuses a
larger one). The reciprocal-lattice route has no such cut but exists only at
the lattice vectors, spaced 2 pi / L, so near Q = 0 a shell holds few of
them, and its cost grows as q_max^3. ``tests/test_md_scattering.py``
compares the two on a 1 794-atom disordered O/Si model (a Matern type-I
hard-core process with an exact g(r)) in a 30 Å box, r_max = 14.9 Å,
r_window 'none', over 1.0 <= Q < 4.0 Å^-1 in 0.1 Å^-1 shells of 132-2 142
vectors: every shell's difference lies within 4 standard errors of the
reciprocal-lattice value (its spread over the shell's vectors / sqrt(n / 2)),
with an RMS below 1.5 of them. Over 20 seeds the worst shell was 3.11
standard errors (a partial; 2.54 for the totals) and the worst RMS 1.17. On
an 8 109-atom model in a 50 Å box (r_max = 24.98 Å) the two agreed within
1.75 standard errors from 0.36 to 4 Å^-1. Below about 1 Å^-1 in a 30 Å box
the shells hold a few dozen vectors each; the comparison is not made there.

WHAT THE SINE ROUTE RESOLVES
----------------------------
Four properties of the sine route follow from its grid and its cut.
:func:`sine_route_notes` states the first three with their values for the
grid in use, and :func:`analyse_trajectory` puts those statements in its
notes; the fourth depends on the values and is noted where it occurs.

* **The cut at r_max.** At finite Q the route's F(Q) = Q [S(Q) - 1] is the
  model's F convolved with (1/pi) C(Q - q), C(x) = Int_0^r_max M(r) cos(x r)
  dr: for 'none' that is sin(x r_max) / (pi x), first zero at pi / r_max,
  full width at half height 3.791 / r_max (sin u / u = 1/2 at u = 1.8955);
  'Lorch' widens it and damps its side lobes. Near Q = 0 the response is the
  window's 3-D transform K(Q, 0) = 4 pi Int_0^r_max r^2 M(r) sin(Qr)/(Qr) dr,
  first zero at 4.493 / r_max for 'none' (tan u = u). Both widths are
  computed from the window and the grid, never typed in, and a test holds
  them to these closed forms. An FSDP width measured on this route includes
  the convolution. With the rectangle sum and linear deposition, 'none' is
  in effect a cut at r_max + dr/2 tapered over one bin.
* **The q = 0 coefficient (a closed box of N atoms).** Like pairs are
  normalised by N_a N_b / V, as pdf.py normalises them. In a closed box the
  pair function tends at large r to 1 - <dN_a dN_b> / (N_a N_b), the
  closed-system term of Lebowitz and Percus [14] (for S(Q) from molecular
  dynamics, Salacuse et al. [15]): -1/N_a for like pairs of an ideal gas,
  about 0 for a dense model whose number fluctuations are small. Written as
  a Fourier series over the cell, the sine route is the box's own S(q) at
  its lattice vectors smoothed by K(Q, q) / V, with the q = 0 point at its
  closed-box value (no number fluctuation, S = 1 - <f^2>/<f>^2 for a
  total). Normalising like pairs by N_a (N_a - 1) instead, as LAMMPS'
  compute rdf and several glass codes do, places that point at the
  ideal-gas value and adds (delta_ab / c_a) K(Q, 0) / V to S_ab - 1 and
  (<f^2>/<f>^2) K(Q, 0) / V to a total, up to terms of relative order
  1/N_a (the exact like-pair term is [N_a / (N_a - 1)] K(Q, 0) / (c_a V)
  + [S_aa(Q) - 1] / (N_a - 1)). Measured 2026-10-07, r_window
  'Lorch', Q from 0.02 to 0.3 Å^-1: on a jittered simple-cubic lattice of
  1 728 atoms of one element, whose lattice-route S(q) is 0.002-0.004 at
  0.21-0.42 Å^-1, this route gives -0.038 to 0.014 and the N_a (N_a - 1)
  form 0.045 to 0.117 (a test holds this); on a 3 000-atom SiO2 glass (SHIK
  potential, neutron), lattice route 0.12-0.21 at 0.18-0.44 Å^-1, this route
  0.133-0.144 and the other form 0.158-0.295. On a dilute model (a Matern
  hard-core process at a packing fraction of 0.035) the other form lies
  nearer: an outside check averaged 150 seeds and found S_OO - exact =
  -0.132 +/- 0.031 at Q = 0.25 Å^-1 with this normalisation and -0.018 +/-
  0.031 with N_a (N_a - 1). Which one a model follows is set by its own
  compressibility, so neither is applied as a correction; the size of the
  term at the grid's first Q is stated, and :func:`sq_from_positions` gives
  the box's own values from 2 pi / L up. In real space the same term is a
  straight line in G(r), between 0 and -4 pi r rho0 <f^2> / (N <f>^2) (the
  ideal-gas end); G(r) is left as the box's pair function, which is also
  what ``pdf.pair_distribution`` gives for a crystal supercell.
* **The grid.** Linear deposition is a convolution with a triangle of
  half-width dr, so the pair part of S(Q) - 1 is multiplied by about
  sinc^2(Q dr / 2) = [sin(Q dr/2) / (Q dr/2)]^2: 0.9948 at Q = 25 Å^-1 for
  dr = 0.01 Å, 0.876 for dr = 0.05 Å. The law holds on average over the
  pairs, not pair by pair: a test fits it within 1 % against a grid-free sum
  over the pairs of a 1 794-atom model at dr = 0.05 Å, and an outside check
  on a 3 000-atom glass measured residues of 5e-6 to 6e-4 once the law was
  applied (dr = 0.005-0.05 Å). The rectangle sum on r_k = k dr is periodic
  in Q with period 2 pi / dr and odd about pi / dr, so above Q = pi / dr it
  returns the mirror image of a lower Q, and a Qmax above pi / dr turns the
  sampled termination kernel into a multiple of the identity (2 G(r) at
  Qmax = 2 pi / dr). Every function here refuses a Q grid or a Qmax beyond
  pi / dr; :func:`real_space_from_sq` likewise refuses r beyond pi / dQ.
* **The lower bound.** A Faber-Ziman total is 1 + (|sum_j f_j e^{iq.r_j}|^2
  / N - <f^2>) / <f>^2, so S(Q) >= 1 - <f^2>/<f>^2 at every Q. A value
  below it comes from the window and the cut, and
  :func:`total_structure_factor` notes every Q where it occurs.

SCATTERING LENGTHS: FROM GEMMI, AND NEVER ZERO BY DEFAULT
---------------------------------------------------------
:func:`scattering_lengths` reads gemmi's tables for the element symbol as
the frame holds it (``md_model.validate_symbol`` has already checked it),
with the same Gaussian sum as ``diffraction.form_factor``. It does not go
through ``diffraction.form_factor`` itself, because that function first
passes the symbol through ``elements.normalise``, which (2026-10-07) turns 15
symbols that are absent from FACET's element tables into a one-letter
element: He -> H, Ne -> N, Kr -> K, Cm -> C, Bk -> B, Cf -> C and nine
heavier ones, so that He was scattered with b(H) = -3.739 fm instead of
3.26 fm. For every other element the two give the same factor, and a test
checks that over the whole table, so a crystal's G(r) here still equals
``pdf.pair_distribution``; for those 15 symbols pdf.py and diffraction.py
keep the one-letter factor (reported, not changed here).

Neutrons: gemmi's ``neutron92`` table of bound coherent scattering lengths
in fm, from Sears [10], for the natural isotopic mixture and real only: gemmi
stores no isotope other than D and no imaginary part. gemmi returns 0 for an
element it has no length for (Po, At, Rn, Fr, Ac, Pu and every element from
Bk on, in gemmi 0.7.1); here such an element is refused with 'TODO: need
reference', never scattered with b = 0. An isotope, or any other length,
is given with ``neutron_lengths_fm`` (element -> b in fm) and
``lengths_source`` (where the numbers come from, required with them); both
are stored in the notes and the provenance, so an 11B-enriched borate is
not compared with natural boron's 5.30 fm without a trace. For the seven
natural elements whose coherent length the NIST table [16] (from Sears)
gives as complex, B, Cd, In, Sm, Eu, Gd and Dy, a note states the imaginary
part, the absorption that is not modelled here. Every result names the
lengths it used.

X-rays: gemmi's ``it92`` four-Gaussian coefficients (International Tables
Vol. C) at (sin theta / lambda)^2 = (Q / 4 pi)^2; the range over which they
were fitted is not recorded in FACET or in gemmi's documentation (reference
to verify), and a note gives the largest sin theta / lambda used. Their
constant term makes some factors reach 0 and go below it at high Q (B from
Q = 38.2 Å^-1, N from 40.0 Å^-1), which no atom's form factor does; a note
names each element and the Q where it happens on the grid in use.
Electrons: gemmi's ``c4322`` coefficients, which gemmi documents as
International Tables Vol. C (2011) table 4.3.2.2, for s up to 2.0 Å^-1;
above Q = 4 pi x 2.0 = 25.13 Å^-1 a note says the curve is extrapolated. An
element without a table is refused for X-rays and electrons too (gemmi has
none from Es on). Units of the weighted functions: barn for neutrons (b in
fm, 1 barn = 100 fm^2 [3]), electron units squared for X-rays, Å^2 for
electrons. Peterson and Keen [3] tabulate <b^2> and <b>^2 for SiO2, MnO and
BaTiO3 from Sears' lengths; a test recovers their values from these
factors.

Only the coherent, elastic, single-scattering part is computed: no Compton
scattering, no anomalous dispersion, no multiple scattering, no instrument
resolution beyond Qmax, Qmin and the window. A measured S(Q) is compared after
its own reduction has removed the rest.

THE FIRST SHARP DIFFRACTION PEAK
--------------------------------
:func:`fsdp` measures position, height and width of the highest point of
S(Q) inside a Q window the caller gives (no default: where the FSDP sits is
a property of the glass). Position and height are the vertex of the parabola
through the highest grid point and its two neighbours. The width is the full
width at half height above a stated baseline, 'zero' (S = 0) or 'minima'
(the straight line through the lowest point on each side within the window),
with the two half-height crossings interpolated linearly between grid
points; a crossing not found inside the window gives NaN and a note, never
an extrapolation, and so does a maximum that rises 0 or less above the
baseline (a negative maximum of a partial over 'zero'). 2 pi / Q_FSDP and
2 pi / FWHM are reported as the repeat distance and the coherence length
that Elliott [11] reads from them.

COMPARISON WITH A MEASURED CURVE
--------------------------------
:func:`compare_with_measured` puts a model curve on a measured one with one
scale factor s and reports Wright's R_chi [12] in the form Zhou et al.
quote it [13]::

    R_chi = sqrt( sum_i [y_meas(x_i) - s y_model(x_i)]^2 / sum_i y_meas(x_i)^2 )

over the measured points x_i inside the range, with the model interpolated
linearly onto them. s is fitted by least squares on the same points
(s = sum y_meas y_model / sum y_model^2) unless the caller gives it. The value
depends on which function is compared (S(Q), F(Q), G(r), T(r)) and over what
range, so both are stored with it, with the measured file, the axis
(Q in Å^-1 or r in Å, named by the caller), the two grid steps and any
notes the caller passes for the model curve. :func:`read_measured` reads
two columns by the rules of ``diffraction.read_pattern`` (the loader of the
PDF panel: comment lines, comma, tab or space separators, sorted by the
first column), and in addition removes a UTF-8 byte-order mark (Excel's
'CSV UTF-8' writes one, and read_pattern would lose the first point to it),
reads UTF-16, accepts semicolons as separators, and refuses a file written
with decimal commas ('0,50<tab>1,234'), which the comma rule would
otherwise read as four columns of other numbers.

FRAMES
------
:func:`analyse_trajectory` runs one frame at a time and averages every curve
through ``md_stats.Series``: each frame weighs the same, and the spread is the
sample standard deviation across frames (``md_stats.SPREAD_NOTE``). A frame
that cannot be read, or whose box became too small for r_max, is skipped with
its reason in the provenance, never dropped silently.

TIMINGS
-------
Measured 2026-10-06 on Windows 11, Intel i5-13420H, Python 3.11.9, numpy
2.4.6, scipy 1.15.1, on the Step 0 benchmark box (``tools/bench_md.make_box(22)``:
10 648 atoms on a jittered 2.3 Å grid in a 50.6 Å cube, Si/O/O/Na at random;
only the size means anything), pinned to the P-cores (CPUs 0-7), median
[min-max] of 5 runs after one untimed run (3 for the lattice sum). The
machine was shared: total CPU load read 100 % before and after, from other
sessions' benchmarks and test suites, so these are figures under load. ::

    frame_partials, r_max 10 Å, dr 0.01 Å          3.05 s [3.01-3.06]
      of which bulk.iter_pairs (3 728 344 pairs,
      boxsize method, 8 blocks)                    2.31 s [2.29-2.35]
      of which glass.partial_rdf                   0.77 s [0.75-0.78]
    partial_structure_factors, 1 501 Q x 1 000 r,
      6 partials, kernel included                  0.059 s
    total_structure_factor, X-ray                  0.001 s
    real_space_from_partials, boxcar Qmax 25       0.001 s
    _LorchOperator built (4 096 Q x 1 000 r)       0.13 s, once per run
    _LorchOperator applied                         0.003 s per frame
    sq_from_positions, q < 3 Å^-1 (58 836
      vectors), neutron + X-ray                    0.71 s [0.70-0.74]
    analyse_trajectory, 3 radiations, Lorch r and
      Q windows, FSDP, per frame                   3.1 s

so a 100-frame run of this size takes about 5 minutes, nearly all of it the
pair search. Unpinned in the same load, with other sessions' processes on the
same cores, the same steps took 2-4 times as long (frame_partials 8.7-12.7 s,
sq_from_positions 7.5-21 s, analyse_trajectory 7-18 s per frame), and the
lattice sum to q < 5 Å^-1 (273 368 vectors) 18-91 s; the spread is the load,
not the code. Python's tracemalloc counted a peak of 121 MB for
frame_partials and 200 MB for a two-frame analyse_trajectory with three
radiations (numpy allocations included).

Added 2026-10-07 (same box and machine, unpinned, total CPU load 6-31 %,
range over 5 runs after one untimed run; each is paid once per run except
the last)::

    sine_route_notes, 1 000 r x 2 001 widths points     0.14-1.1 s
    _sq_to_g_matrix (pdf_g_from_sq), 1 000 r x 834 Q    0.08-0.64 s
    pdf_g_from_sq applied, per frame and radiation      0.001 s
    scattering_lengths, X-ray, 4 elements x 1 501 Q     0.001-0.005 s

REFERENCES
----------
[1] D. A. Keen, "A comparison of various commonly used correlation functions
    for describing total scattering", *Journal of Applied Crystallography* 34
    (2001) 172-177, https://doi.org/10.1107/S0021889800019993
[2] P. F. Peterson, D. Olds, M. T. McDonnell and K. Page, "Illustrated
    formalisms for total scattering data: a guide for new practitioners",
    *Journal of Applied Crystallography* 54 (2021) 317-332,
    https://doi.org/10.1107/S1600576720015630
[3] P. F. Peterson and D. A. Keen, "Illustrated formalisms for total
    scattering data: a guide for new practitioners. Corrigendum and
    addendum", *Journal of Applied Crystallography* 54 (2021) 1542-1545,
    https://doi.org/10.1107/S1600576721007664
[4] T. E. Faber and J. M. Ziman, "A theory of the electrical properties of
    liquid metals III. The resistivity of binary alloys", *Philosophical
    Magazine* 11 (1965) 153-173, https://doi.org/10.1080/14786436508211931
[5] E. Lorch, "Neutron diffraction by germania, silica and radiation-damaged
    silica glasses", *Journal of Physics C: Solid State Physics* 2 (1969)
    229-237, https://doi.org/10.1088/0022-3719/2/2/305
[6] A. B. Bhatia and D. E. Thornton, "Structural aspects of the electrical
    resistivity of binary alloys", *Physical Review B* 2 (1970) 3004,
    https://doi.org/10.1103/PhysRevB.2.3004
[7] P. S. Salmon, "The structure of molten and glassy 2:1 binary systems: an
    approach using the Bhatia-Thornton formalism", *Proceedings of the Royal
    Society of London A* 437 (1992) 591-606,
    https://doi.org/10.1098/rspa.1992.0081
[8] N. W. Ashcroft and D. C. Langreth, "Structure of binary liquid
    mixtures. I", *Physical Review* 156 (1967) 685-692,
    https://doi.org/10.1103/PhysRev.156.685
[9] H. E. Fischer, A. C. Barnes and P. S. Salmon, "Neutron and x-ray
    diffraction studies of liquids and glasses", *Reports on Progress in
    Physics* 69 (2006) 233-299, https://doi.org/10.1088/0034-4885/69/1/R05
[10] V. F. Sears, "Neutron scattering lengths and cross sections", *Neutron
    News* 3 (1992) 26-37, https://doi.org/10.1080/10448639208218770
[11] S. R. Elliott, "Medium-range structural order in covalent amorphous
    solids", *Nature* 354 (1991) 445-452, https://doi.org/10.1038/354445a0
[12] A. C. Wright, "The comparison of molecular dynamics simulations with
    diffraction experiments", *Journal of Non-Crystalline Solids* 159 (1993)
    264-268, https://doi.org/10.1016/0022-3093(93)90232-M
[13] Q. Zhou, T. Du, L. Guo, M. M. Smedskjaer and M. Bauchy, "New insights
    into the structure of sodium silicate glasses by force-enhanced atomic
    refinement", *Journal of Non-Crystalline Solids* 536 (2020) 120006,
    https://doi.org/10.1016/j.jnoncrysol.2020.120006 (preprint
    arXiv:1910.13996, Eq. 2, where the formula was read)
[14] J. L. Lebowitz and J. K. Percus, "Long-range correlations in a closed
    system with applications to nonuniform fluids", *Physical Review* 122
    (1961) 1675-1691, https://doi.org/10.1103/PhysRev.122.1675
[15] J. J. Salacuse, A. R. Denton and P. A. Egelstaff, "Finite-size effects
    in molecular dynamics simulations: static structure factor and
    compressibility. I. Theoretical method", *Physical Review E* 53 (1996)
    2382-2389, https://doi.org/10.1103/PhysRevE.53.2382
[16] NIST Center for Neutron Research, "Neutron scattering lengths and cross
    sections", https://www.ncnr.nist.gov/resources/n-lengths/list.html (the
    data of Sears [10]; read 2026-10-07)
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bulk, diffraction, glass, pdf
from .md_model import Frame, FrameError
from .md_stats import Provenance, Scalar, Series, check_fractions

__all__ = [
    "RADIATIONS", "R_WINDOWS", "Q_WINDOWS", "FSDP_BASELINES", "TODO_REFERENCE",
    "BARN_PER_FM2", "ELECTRON_S_MAX_INV_ANG", "KERNEL_VALUES_LIMIT",
    "MODES_VALUES_LIMIT", "WEIGHT_UNITS", "R_CHI_DEFINITION",
    "NEUTRON_IMAGINARY_FM", "MEASURED_AXES",
    "scattering_lengths", "faber_ziman_weights",
    "FramePartials", "frame_partials", "partial_structure_factors",
    "sine_route_notes",
    "TotalSQ", "total_structure_factor",
    "RealSpace", "real_space_from_partials", "real_space_from_sq",
    "BhatiaThornton", "bhatia_thornton",
    "density_modes", "DirectSQ", "sq_from_positions",
    "FSDP", "fsdp",
    "Measured", "read_measured", "Comparison", "compare_with_measured",
    "TotalScattering", "ScatteringResult", "analyse_trajectory",
]

# ---------------------------------------------------------------------------
# constants: physical ones say where they come from, the rest are choices
# ---------------------------------------------------------------------------

RADIATIONS = diffraction.RADIATIONS            # ("X-ray", "neutron", "electron")
R_WINDOWS = ("none", "Lorch")                  # M(r) of the r -> Q transform
Q_WINDOWS = ("boxcar", "Lorch")                # the Qmax window, as pdf.py
FSDP_BASELINES = ("zero", "minima")
TODO_REFERENCE = "TODO: need reference"

# 1 barn = 1e-28 m^2 = 100 fm^2: the definition of the barn (Peterson and
# Keen [3], Table 3 note). gemmi's neutron lengths are in fm.
BARN_PER_FM2 = 0.01

# gemmi documents its c4322 electron coefficients as tabulated for
# s = sin(theta)/lambda up to 2.0 Å^-1 (gemmi.readthedocs.io, "scattering").
ELECTRON_S_MAX_INV_ANG = 2.0

# The unit of f_a f_b, per radiation, for the functions that are not divided
# by <f>^2 (F_K, G_K, D, T).
WEIGHT_UNITS = {"neutron": "barn", "X-ray": "electrons^2", "electron": "Å^2"}

# The most values the cached sine kernel (n_Q x n_r) may hold. A memory
# choice, not a physical value: 25 million doubles is 200 MB.
KERNEL_VALUES_LIMIT: int = 25_000_000

# The most complex values the reciprocal-lattice route keeps per axis table
# (N atoms x (2 H + 1) indices). A memory choice: 20 million complex values is
# 320 MB per table, two tables.
MODES_VALUES_LIMIT: int = 20_000_000

# Natural elements whose bound coherent neutron scattering length the NIST
# table [16] (Sears [10]) gives as complex: the imaginary part in fm, which
# describes absorption and which gemmi's neutron92 does not store. Read from
# https://www.ncnr.nist.gov/resources/n-lengths/list.html on 2026-10-07
# (B 5.30-0.213i, Cd 4.87-0.70i, In 4.065-0.0539i, Sm 0.80-1.65i,
# Eu 7.22-1.26i, Gd 6.5-13.82i, Dy 16.9-0.276i). Used for a note only.
NEUTRON_IMAGINARY_FM = {"B": -0.213, "Cd": -0.70, "In": -0.0539, "Sm": -1.65,
                        "Eu": -1.26, "Gd": -13.82, "Dy": -0.276}

# The axes a measured curve can carry, named with their units.
MEASURED_AXES = ("q_inv_ang", "r_ang")

# How many points sample the window's transforms when their widths are
# measured (sine_route_notes), over 0 <= Q <= 4 pi / r_max. A numerical
# choice: the step, pi / (500 r_max), is about 1/600 of the narrowest width
# measured (3.79 / r_max), and the crossings are interpolated linearly
# between its points.
_WIDTH_POINTS = 2001

R_CHI_DEFINITION = (
    "R_chi = sqrt( sum_i [y_meas(x_i) - s y_model(x_i)]^2 / "
    "sum_i y_meas(x_i)^2 ), over the measured points x_i in the range, the "
    "model interpolated linearly onto x_i, s the scale factor (Wright, J. "
    "Non-Cryst. Solids 159 (1993) 264, in the form of Zhou et al., J. "
    "Non-Cryst. Solids 536 (2020) 120006, Eq. 2)")


# ---------------------------------------------------------------------------
# small checks
# ---------------------------------------------------------------------------

def _positive(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out) or out < 0.0 or (out == 0.0 and not allow_zero):
        need = "0 or more" if allow_zero else "above 0"
        raise ValueError(f"{name} is {value!r}; a finite number {need} is "
                         "needed")
    return out


def _number(value, name: str) -> float:
    """A finite float; ValueError naming the argument otherwise."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} is {value!r}; a finite number is needed")
    return out


def _range_pair(value, name: str) -> tuple[float, float]:
    """Two finite numbers (low, high) with low < high."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} needs two numbers (low, high), not a text")
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"{name} needs two numbers (low, high)") from None
    if len(items) != 2:
        raise ValueError(f"{name} needs two numbers (low, high), not "
                         f"{len(items)}")
    low, high = (_number(v, name) for v in items)
    if not low < high:
        raise ValueError(f"{name} ({low:g}, {high:g}) needs low < high")
    return low, high


def _radiation(value) -> str:
    if value not in RADIATIONS:
        raise ValueError(f"radiation {value!r}: one of {RADIATIONS}")
    return value


def _radiation_list(values, name: str = "radiations") -> tuple[str, ...]:
    """A sequence of radiation names; a bare text is refused, not split."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} needs a sequence of names, such as "
                         "('neutron',) or ('neutron', 'X-ray'); a bare text "
                         "would be read letter by letter")
    try:
        items = list(values)
    except TypeError:
        raise ValueError(f"{name} needs a sequence of names from "
                         f"{RADIATIONS}") from None
    return tuple(dict.fromkeys(_radiation(item) for item in items))


def _choice(value, name: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} {value!r}: one of {allowed} (there is no "
                         "default; the choice changes the result)")
    return value


def _axis(values, name: str, *, nonnegative: bool = True) -> np.ndarray:
    """A 1-D, finite, strictly increasing grid (>= 0 when asked)."""
    try:
        out = np.array(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name}: the values are not numbers") from None
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name}: a 1-D array of at least one value is needed")
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: a value is NaN or infinite")
    if nonnegative and (out < 0.0).any():
        raise ValueError(f"{name}: a value is negative")
    if out.size > 1 and not (np.diff(out) > 0).all():
        raise ValueError(f"{name}: the values have to increase strictly")
    out.setflags(write=False)
    return out


def _q_weights(values: np.ndarray) -> np.ndarray:
    """Quadrature weights of a sum over a grid: each point's local spacing.

    Interior points carry half the distance between their two neighbours,
    the two ends the step next to them. On a uniform grid every weight is
    the one step, so the sum is ``pdf.inverse_transform``'s rectangle rule;
    on a grid that is uniform only to the digits a file printed, or not at
    all (a grid converted from 2-theta), each point still carries its own
    spacing.
    """
    weights = np.empty_like(values)
    weights[1:-1] = 0.5 * (values[2:] - values[:-2])
    weights[0] = values[1] - values[0]
    weights[-1] = values[-1] - values[-2]
    return weights


def _nyquist_q(q_value: float, dr: float, name: str) -> None:
    """Refuse a Q beyond pi / dr, where the r-grid sum mirrors lower Q."""
    limit = math.pi / dr
    if q_value > limit * (1.0 + 1e-12):
        raise ValueError(
            f"{name} reaches {q_value:.6g} Å^-1, beyond pi / dr = "
            f"{limit:.6g} Å^-1 for dr = {dr:g} Å: a sum over the grid "
            "r_k = k dr is periodic in Q with period 2 pi / dr and odd about "
            "pi / dr, so above it every value mirrors the value at a lower "
            f"Q. A dr_ang of at most pi / {q_value:.6g} = "
            f"{math.pi / q_value:.4g} Å, or a lower Q, is needed")


def _frozen(array) -> np.ndarray:
    out = np.array(array, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


def _pair_name(pair: tuple[str, str]) -> str:
    return f"{pair[0]}-{pair[1]}"


def _unordered_pairs(elements: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for k, a in enumerate(elements) for b in elements[k:]]


# ---------------------------------------------------------------------------
# scattering lengths and weights
# ---------------------------------------------------------------------------

def _gemmi_element(symbol: str):
    """gemmi's element for exactly this symbol, or None.

    No relabelling: 'He' is helium (``elements.normalise`` would give H),
    and a symbol gemmi does not know, or knows under another spelling, is
    None.
    """
    import gemmi

    if not isinstance(symbol, str) or not symbol:
        return None
    try:
        element = gemmi.Element(symbol)
    except Exception:          # gemmi raises its own types for unknown symbols
        return None
    if not element.atomic_number or element.name != symbol:
        return None
    return element


def _table_missing(symbol: str, radiation: str) -> bool:
    """True when gemmi holds no factor for this element and radiation."""
    element = _gemmi_element(symbol)
    if element is None:
        return True
    if radiation == "neutron":
        coefs = list(element.neutron92.get_coefs())
        return not coefs or float(coefs[0]) == 0.0
    table = element.c4322 if radiation == "electron" else element.it92
    return table is None


def _factor(symbol: str, radiation: str, stol2):
    """gemmi's factor of this exact element at (sin theta / lambda)^2.

    The expression of ``diffraction.scattering_coefficients`` and
    ``diffraction.form_factor`` (sum_i a_i exp(-b_i stol2) + c; a neutron
    length as one term with b = 0), read from the element as given.
    """
    element = _gemmi_element(symbol)
    if radiation == "neutron":
        a = np.array([float(element.neutron92.get_coefs()[0])])
        b = np.zeros(1)
        c = 0.0
    elif radiation == "electron":
        table = element.c4322
        a = np.array(list(table.a), float)
        b = np.array(list(table.b), float)
        c = 0.0
    else:
        table = element.it92
        a = np.array(list(table.a), float)
        b = np.array(list(table.b), float)
        c = float(table.c)
    s2 = np.asarray(stol2, float)
    total = np.full(s2.shape, c, float)
    for coefficient, exponent in zip(a, b):
        total = total + coefficient * np.exp(-exponent * s2)
    return total if total.shape else float(total)


def _neutron_overrides(lengths_fm, source, symbols) -> dict[str, float]:
    """The user's neutron lengths, checked; {} when none were given."""
    if lengths_fm is None:
        if source is not None:
            raise ValueError("lengths_source was given without "
                             "neutron_lengths_fm")
        return {}
    if not isinstance(lengths_fm, Mapping) or not lengths_fm:
        raise ValueError("neutron_lengths_fm needs a mapping of element "
                         "symbol to a bound coherent length in fm, such as "
                         "{'B': 6.65}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "neutron_lengths_fm needs lengths_source: where the lengths come "
            "from (the isotope and the table, such as '11B, Sears 1992'); it "
            "is stored with every result that uses them")
    out: dict[str, float] = {}
    for key, value in lengths_fm.items():
        if _gemmi_element(key) is None:
            raise ValueError(f"neutron_lengths_fm: {key!r} is not an element "
                             "symbol of gemmi's periodic table")
        out[key] = _number(value, f"neutron_lengths_fm[{key!r}]")
    absent = sorted(set(out) - set(symbols))
    if absent:
        raise ValueError(f"neutron_lengths_fm names {', '.join(absent)}, "
                         "which the model does not hold")
    return out


_TABLE_NAMES = {
    "neutron": "coherent neutron scattering length (gemmi neutron92, Sears "
               "1992)",
    "X-ray": "X-ray form factor (gemmi it92)",
    "electron": "electron scattering factor (gemmi c4322)",
}


def scattering_lengths(elements: Iterable[str], radiation: str,
                       q_inv_ang=None, *,
                       neutron_lengths_fm: Mapping[str, float] | None = None,
                       lengths_source: str | None = None
                       ) -> dict[str, np.ndarray | float]:
    """Scattering factor per element: a float at Q = 0, or an array over Q.

    Neutron: the bound coherent length in fm (no Q dependence), from gemmi's
    neutron92 for the natural isotopic mixture unless ``neutron_lengths_fm``
    gives it (then ``lengths_source``, saying where the numbers come from, is
    required; elements it does not name keep gemmi's length). X-ray: f in
    electrons at (sin theta / lambda)^2 = (Q / 4 pi)^2. Electron: f in Å.
    The symbols are read as given ('He' is helium; see the module
    docstring for why ``diffraction.form_factor`` is not called). An element
    with no entry in gemmi's table for that radiation is refused with
    ValueError naming it and 'TODO: need reference'; it is never given 0.
    """
    radiation = _radiation(radiation)
    symbols = sorted({str(s) for s in elements})
    if not symbols:
        raise ValueError("no elements were given")
    unknown = [s for s in symbols if _gemmi_element(s) is None]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)}: not an element symbol of gemmi's periodic "
            f"table, so FACET holds no {_TABLE_NAMES[radiation]} for it; "
            f"{TODO_REFERENCE}")
    given = _neutron_overrides(neutron_lengths_fm, lengths_source, symbols)
    if radiation != "neutron":
        given = {}
    missing = [s for s in symbols if s not in given
               and _table_missing(s, radiation)]
    if missing:
        raise ValueError(
            f"{radiation} scattering: FACET's data hold no "
            f"{_TABLE_NAMES[radiation]} for {', '.join(missing)}; "
            f"{TODO_REFERENCE}. The calculation is refused rather than run "
            "with a factor of 0"
            + ("; a length can be given with neutron_lengths_fm and "
               "lengths_source" if radiation == "neutron" else "") + ".")
    if q_inv_ang is None:
        return {s: given[s] if s in given else float(_factor(s, radiation,
                                                              0.0))
                for s in symbols}
    q = np.asarray(q_inv_ang, dtype=np.float64)
    stol2 = (q / (4.0 * math.pi)) ** 2
    out = {}
    for s in symbols:
        values = given[s] if s in given else _factor(s, radiation, stol2)
        out[s] = np.broadcast_to(np.asarray(values, dtype=np.float64),
                                 q.shape).copy()
    return out


def _factor_notes(symbols: Sequence[str], radiation: str,
                  given: Mapping[str, float], source: str | None
                  ) -> list[str]:
    """Which factor each element was given, and from where."""
    symbols = sorted(symbols)
    if radiation == "neutron":
        parts = [f"{s} {given[s]:.6g} (given)" if s in given else
                 f"{s} {_factor(s, 'neutron', 0.0):.6g}" for s in symbols]
        text = "neutron bound coherent lengths in fm: " + ", ".join(parts)
        natural = [s for s in symbols if s not in given]
        if natural:
            text += (f"; {', '.join(natural)} from gemmi neutron92 (Sears "
                     "1992), natural isotopic abundance, real part")
        if given:
            text += f"; the given lengths from: {source}"
        notes = [text]
        for s in natural:
            if s in NEUTRON_IMAGINARY_FM:
                real = _factor(s, "neutron", 0.0)
                imag = NEUTRON_IMAGINARY_FM[s]
                notes.append(
                    f"natural {s}: the NIST table (Sears 1992) gives its "
                    f"coherent length as {real:g} - {abs(imag):g}i fm; the "
                    "imaginary part describes absorption, which is not "
                    f"modelled here, and only the real part {real:g} fm "
                    "enters")
        return notes
    table, unit = (("it92", "electrons") if radiation == "X-ray" else
                   ("c4322", "Å"))
    return [f"{radiation} factors (gemmi {table}) taken at each Q; at Q = 0, "
            f"in {unit}: " + ", ".join(
                f"{s} {_factor(s, radiation, 0.0):.5g}" for s in symbols)]


def faber_ziman_weights(concentrations: Mapping[str, float],
                        lengths: Mapping[str, np.ndarray | float]):
    """Faber-Ziman weights of the unordered element pairs, and <f>, <f^2>.

    ``w_ab = (2 - delta_ab) c_a c_b f_a f_b / <f>^2``, so the weights of the
    unordered pairs sum to 1 (at every Q when the factors are arrays).
    Returns ``(weights, f_mean, f2_mean)``. ValueError when the
    concentrations are not a fraction set (``md_stats.check_fractions``), an
    element has no factor, or <f> is zero (a neutron composition whose
    lengths cancel), where the normalisation is undefined.
    """
    symbols = sorted(concentrations)
    check_fractions([concentrations[s] for s in symbols], "concentrations")
    absent = [s for s in symbols if s not in lengths]
    if absent:
        raise ValueError(f"no scattering factor was given for {absent}")
    c = {s: float(concentrations[s]) for s in symbols}
    f = {s: np.asarray(lengths[s], dtype=np.float64) for s in symbols}
    f_mean = sum(c[s] * f[s] for s in symbols)
    f2_mean = sum(c[s] * f[s] * f[s] for s in symbols)
    if (np.abs(f_mean) < 1e-30).any():
        raise ValueError(
            "the mean scattering factor <f> = sum c_a f_a of this composition "
            "is 0, so the Faber-Ziman normalisation (division by <f>^2) is "
            "undefined; this happens for a neutron composition whose "
            "scattering lengths cancel")
    weights = {}
    for a, b in _unordered_pairs(symbols):
        share = c[a] * c[b] * f[a] * f[b] / (f_mean * f_mean)
        weights[(a, b)] = share if a == b else 2.0 * share
    return weights, f_mean, f2_mean


def _unit_factor(radiation: str) -> float:
    """Multiplies f_a f_b into the weight unit (fm^2 -> barn for neutrons)."""
    return BARN_PER_FM2 if radiation == "neutron" else 1.0


def _range_notes(radiation: str, q: np.ndarray,
                 lengths: Mapping[str, np.ndarray] | None = None
                 ) -> list[str]:
    """Where Q goes past what the tables document, and where a fitted
    X-ray or electron factor reaches 0 or below on this grid."""
    if not q.size:
        return []
    notes = []
    s_max = float(q.max()) / (4.0 * math.pi)
    if radiation == "electron" and s_max > ELECTRON_S_MAX_INV_ANG:
        notes.append(
            f"electron factors used to sin(theta)/lambda = {s_max:.4g} "
            f"Å^-1 (Q = {float(q.max()):.4g} Å^-1); gemmi documents its "
            f"c4322 coefficients for s up to {ELECTRON_S_MAX_INV_ANG} "
            f"Å^-1, so above Q = {4.0 * math.pi * ELECTRON_S_MAX_INV_ANG:.4g} "
            "Å^-1 they are extrapolated")
    if radiation == "X-ray":
        notes.append(
            f"X-ray form factors used to sin(theta)/lambda = {s_max:.4g} "
            f"Å^-1 (Q = {float(q.max()):.4g} Å^-1); the range over which "
            "the it92 coefficients were fitted is not recorded in FACET "
            "(reference to verify)")
    if lengths is not None and radiation != "neutron":
        for symbol in sorted(lengths):
            values = np.broadcast_to(np.asarray(lengths[symbol], float),
                                     q.shape)
            low = np.flatnonzero(values <= 0.0)
            if low.size:
                notes.append(
                    f"the {radiation} factor of {symbol} is 0 or below from "
                    f"Q = {float(q[low[0]]):.4g} Å^-1 on this grid ({low.size} "
                    f"of {q.size} Q points; {float(values[low].min()):.4g} at "
                    "the lowest): an atom's form factor is positive, so "
                    "there the fitted Gaussian sum is outside the range it "
                    "describes")
    return notes


# ---------------------------------------------------------------------------
# partial pair distribution functions from one frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class FramePartials:
    """One frame's partial pair functions on pdf.pair_distribution's r grid.

    ``g[(a, b)]``, a <= b alphabetically: g_ab(r) from ``glass.partial_rdf``.
    ``pair_hist[(a, b)]``, every ordered pair of elements: the deposited count
    of pairs with an ``a`` atom at the centre and a ``b`` neighbour
    (``pdf._deposit``, unit weights), recovered from g as
    g N_a N_b 4 pi r^2 dr / V, so its sum is the number of such pairs within
    reach of the grid. ``n_cum[(a, b)]``: the running coordination N_ab(r),
    the mean number of b atoms within r of an a atom (``glass.partial_rdf``).
    """

    r_ang: np.ndarray
    dr_ang: float
    elements: tuple[str, ...]
    counts: Mapping[str, int]
    n_atoms: int
    volume_ang3: float
    pair_hist: Mapping[tuple[str, str], np.ndarray]
    g: Mapping[tuple[str, str], np.ndarray]
    n_cum: Mapping[tuple[str, str], np.ndarray]
    search_r_ang: float
    d_min_ang: float
    n_below_d_min: int
    search_method: str
    timestep: int | None = None
    time_ps: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def r_max_ang(self) -> float:
        return float(self.r_ang[-1])

    @property
    def rho0_per_ang3(self) -> float:
        return self.n_atoms / self.volume_ang3

    @property
    def concentrations(self) -> dict[str, float]:
        return {s: self.counts[s] / self.n_atoms for s in self.elements}

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.g)


def _r_grid(r_max_ang, dr_ang) -> tuple[np.ndarray, float]:
    r_max = _positive(r_max_ang, "r_max_ang")
    dr = _positive(dr_ang, "dr_ang")
    if dr >= r_max:
        raise ValueError(f"dr_ang {dr} is not below r_max_ang {r_max}")
    r = np.arange(dr, r_max + 0.5 * dr, dr)       # pdf.pair_distribution's grid
    r.setflags(write=False)
    return r, dr


def frame_partials(frame: Frame, pairs: Iterable[bulk.PairTable] | None = None,
                   *, r_max_ang: float, dr_ang: float) -> FramePartials:
    """Partial g_ab(r) of one frame, from the frame's one pair search.

    ``pairs``: the blocks of ``bulk.iter_pairs`` on this frame (shared with
    another analysis of the same frame), searched to at least r_max + dr,
    because a distance deposits into the grid points on both sides of it.
    None runs that search here, with d_min_ang = 0, so no pair other than an
    atom with itself is left out. The grid is ``pdf.pair_distribution``'s,
    ``arange(dr, r_max + dr/2, dr)``; ``dr_ang`` is a bin width, a method
    choice with no default here.

    ValueError when r_max + dr exceeds half the smallest perpendicular box
    width (beyond it an atom meets its own periodic images and the
    correlations are those of the box), and for pair blocks that are of
    another frame, do not cover every atom once, or do not reach far enough.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    r, dr = _r_grid(r_max_ang, dr_ang)
    reach = float(r[-1]) + dr
    half = float(frame.perpendicular_widths_ang.min()) / 2.0
    if reach > half:
        raise ValueError(
            f"r_max + dr = {reach:.6g} Å is beyond half the smallest "
            f"perpendicular width of the box ({half:.6g} Å); there an atom "
            "meets its own periodic images, so g(r) and its transform would "
            "describe the box rather than the model. A smaller r_max_ang is "
            "needed")
    n = frame.n_atoms
    symbols, element_index = np.unique(frame.elements, return_inverse=True)
    elements = tuple(str(s) for s in symbols)
    counts = {s: int(c) for s, c in zip(elements, np.bincount(
        element_index, minlength=len(elements)))}

    own_search = pairs is None
    if own_search:
        pairs = bulk.iter_pairs(frame, reach, d_min_ang=0.0,
                                vectors_within_ang=0.0)
    seen: list[bulk.PairTable] = []

    def recorded(blocks):
        """The blocks as given, each one noted as glass.partial_rdf takes it."""
        for block in bulk._blocks_of(blocks):
            seen.append(block)
            yield block

    # the deposition and the normalisation are glass.py's, so the glass
    # descriptors and the scattering read one g(r); glass.partial_rdf also
    # checks the blocks' frame, coverage and reach. The refusal above keeps
    # its grid uncut, so every frame has the same grid.
    rdf = glass.partial_rdf(frame, recorded(pairs), r_max_ang=r_max_ang,
                            dr_ang=dr)
    first = next(iter(rdf.values()))
    if first.r_ang.shape != r.shape or not np.array_equal(first.r_ang, r):
        raise ValueError("glass.partial_rdf returned a grid other than "
                         "arange(dr, r_max + dr/2, dr)")
    n_below = sum(block.n_below_d_min for block in seen)
    d_min, method, search_r = (seen[-1].d_min_ang, seen[-1].method,
                               seen[-1].r_ang)

    volume = frame.volume_ang3
    shell = 4.0 * math.pi * r * r * dr
    pair_hist: dict[tuple[str, str], np.ndarray] = {}
    g: dict[tuple[str, str], np.ndarray] = {}
    n_cum: dict[tuple[str, str], np.ndarray] = {}
    extra: list[str] = []
    for (a, b), partial in rdf.items():
        n_cum[(a, b)] = partial.n_cum
        # the deposited count glass.partial_rdf normalised: g N_a N_b shell / V
        pair_hist[(a, b)] = _frozen(partial.g * (counts[a] * float(counts[b]))
                                    * shell / volume)
        if a <= b:
            g[(a, b)] = partial.g
        extra.extend(note for note in partial.notes if note not in extra)

    notes = [f"partial g(r) by glass.partial_rdf on r = {r[0]:.6g} .. "
             f"{r[-1]:.6g} Å, dr = {dr:g} Å (pdf.pair_distribution's grid), "
             "each distance split linearly between its two grid points "
             "(pdf._deposit); pairs from "
             + ("the frame's own search" if own_search else
                "the pair blocks given") + f" to {search_r:.6g} Å"] + extra
    if n_below:
        notes.append(f"{n_below} ordered pairs at or below d_min = {d_min:g} Å "
                     "were left out by the pair search (bulk.iter_pairs) and "
                     "are not in g(r)")
    for symbol in elements:
        if counts[symbol] == 1:
            notes.append(
                f"{symbol} has one atom in this frame: no {symbol}-{symbol} "
                f"pair exists, so g {symbol}-{symbol} is 0 at every r and "
                f"S {symbol}-{symbol} by the sine route is the window's own "
                "transform (1 - 4 pi rho0 Int r^2 M sinc(Qr) dr), not a "
                "correlation")
    return FramePartials(
        r_ang=r, dr_ang=dr, elements=elements, counts=counts, n_atoms=n,
        volume_ang3=volume, pair_hist=pair_hist, g=g, n_cum=n_cum,
        search_r_ang=search_r, d_min_ang=d_min, n_below_d_min=n_below,
        search_method=method, timestep=frame.timestep, time_ps=frame.time_ps,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# the sine route: S_ab(Q)
# ---------------------------------------------------------------------------

def _sine_kernel(q: np.ndarray, r: np.ndarray, dr: float,
                 r_window: str) -> np.ndarray:
    """K[Q, r] = dr r^2 M(r) sin(Qr)/(Qr): the rectangle rule of
    ``pdf.forward_transform``, with the 4 pi rho0 left out."""
    size = q.size * r.size
    if size > KERNEL_VALUES_LIMIT:
        raise ValueError(
            f"the sine kernel would hold {q.size} x {r.size} = {size} values, "
            f"above KERNEL_VALUES_LIMIT ({KERNEL_VALUES_LIMIT}, a memory "
            "choice); a coarser Q grid or a shorter r grid needs fewer")
    return np.sinc(np.outer(q, r) / math.pi) * (
        dr * r * r * _r_window_values(r, r_window))[None, :]


def _r_window_values(r: np.ndarray, r_window: str) -> np.ndarray:
    """M(r): 1 ('none') or sin(pi r / r_max) / (pi r / r_max) ('Lorch')."""
    return np.ones_like(r) if r_window == "none" else np.sinc(r / r[-1])


def _first_below(axis: np.ndarray, curve: np.ndarray, level: float) -> float:
    """First axis value where ``curve`` falls to ``level``, linearly
    interpolated; NaN if it never does (or starts there)."""
    below = np.flatnonzero(curve <= level)
    if not below.size or below[0] == 0:
        return float("nan")
    k = int(below[0])
    t = (curve[k - 1] - level) / (curve[k - 1] - curve[k])
    return float(axis[k - 1] + t * (axis[k] - axis[k - 1]))


def _window_widths(r: np.ndarray, dr: float, r_window: str
                   ) -> dict[str, float]:
    """The sine route's two responses, measured on its own grid sums.

    'conv_fwhm', 'conv_zero': full width at half height and first zero of
    C(x) = sum_k dr M(r_k) cos(x r_k), the kernel that convolves
    F(Q) = Q [S(Q) - 1] at finite Q. 'k0_fwhm', 'k0_zero': the same for
    K(Q, 0) = 4 pi sum_k dr r_k^2 M(r_k) sinc(Q r_k), the response at Q -> 0
    (FWHM taken as twice the Q of half height, the curve being even in Q).
    """
    m = _r_window_values(r, r_window)
    lag_inv_ang = np.linspace(0.0, 4.0 * math.pi / float(r[-1]),
                              _WIDTH_POINTS)
    conv = np.empty_like(lag_inv_ang)
    k0 = np.empty_like(lag_inv_ang)   # r^2 sinc(Q r) = r sin(Q r) / Q, Q > 0
    for start in range(0, lag_inv_ang.size, 256):     # 256 rows at a time
        phase = np.outer(lag_inv_ang[start:start + 256], r)
        conv[start:start + 256] = np.cos(phase) @ (dr * m)
        k0[start:start + 256] = np.sin(phase) @ (dr * r * m)
    k0[1:] /= lag_inv_ang[1:]
    k0[0] = float(np.sum(dr * r * r * m))
    return {"conv_fwhm": 2.0 * _first_below(lag_inv_ang, conv, 0.5 * conv[0]),
            "conv_zero": _first_below(lag_inv_ang, conv, 0.0),
            "k0_fwhm": 2.0 * _first_below(lag_inv_ang, k0, 0.5 * k0[0]),
            "k0_zero": _first_below(lag_inv_ang, k0, 0.0)}


def sine_route_notes(partials: FramePartials, q_inv_ang, *, r_window: str
                     ) -> tuple[str, ...]:
    """What the sine route's grid and cut do to S(Q), with their values.

    Three of the four statements of the module docstring's 'What the sine
    route resolves', measured for this r grid, this window and this Q grid:
    the convolution widths of the cut, the q = 0 coefficient term at the
    grid's first Q, and the attenuation sinc^2(Q dr / 2) at its last Q with
    pi / dr. The fourth, a total below its lower bound, depends on the
    values and is noted by :func:`total_structure_factor` where it occurs.
    A Q grid beyond pi / dr is refused, as in
    :func:`partial_structure_factors`. :func:`analyse_trajectory` adds the
    three to its notes; a caller of :func:`partial_structure_factors` can
    attach them to its own results.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    r_window = _choice(r_window, "r_window", R_WINDOWS)
    _nyquist_q(float(q[-1]), partials.dr_ang, "q_inv_ang")
    return _route_texts(partials, q, r_window)[0]


def _route_texts(partials: FramePartials, q: np.ndarray, r_window: str
                 ) -> tuple[tuple[str, ...], dict[str, float]]:
    """:func:`sine_route_notes` on checked arguments, and the widths it
    measured (so :func:`analyse_trajectory` measures them once)."""
    r, dr = partials.r_ang, partials.dr_ang
    r_end = float(r[-1])
    widths = _window_widths(r, dr, r_window)
    k_first = float((4.0 * math.pi / partials.volume_ang3) * _sine_kernel(
        q[:1], r, dr, r_window).sum())
    q_last = float(q[-1])
    attenuation = float(np.sinc(q_last * dr / (2.0 * math.pi)) ** 2)
    texts = (
        f"sine route on r = {float(r[0]):.6g} .. {r_end:.6g} Å, dr = {dr:g} "
        f"Å, r_window '{r_window}': at finite Q, F(Q) = Q [S(Q) - 1] is the "
        "model's convolved with the window's cosine transform, of full "
        f"width at half height {widths['conv_fwhm']:.4g} Å^-1 and first "
        f"zero {widths['conv_zero']:.4g} Å^-1, so a peak width measured on "
        "it includes that broadening; near Q = 0 the response is the "
        "window's 3-D transform K(Q, 0), of full width at half height "
        f"{widths['k0_fwhm']:.4g} Å^-1 and first zero "
        f"{widths['k0_zero']:.4g} Å^-1, and below that zero the values "
        "carry the cut at r_max and the q = 0 term "
        "(sq_from_positions gives the box's own S(q) from 2 pi / L)",
        "q = 0 term: like pairs are normalised by N_a N_b / V (pdf.py's "
        "convention, the closed box's q = 0 point at no number fluctuation); "
        "normalising them by N_a (N_a - 1) would add (delta_ab / c_a) "
        "K(Q, 0) / V to S_ab - 1 and (<f^2>/<f>^2) K(Q, 0) / V to a total "
        "(up to terms of relative order 1/N_a), "
        f"with K(Q, 0) / V = {k_first:.4g} at Q = {float(q[0]):.4g} Å^-1, "
        "the grid's first point; the model's own value lies between the two "
        "and is set by its compressibility (module docstring)",
        f"the linear deposition on the r grid multiplies the pair part of "
        f"S(Q) - 1 by about sinc^2(Q dr / 2): {attenuation:.4g} at the "
        f"grid's last Q, {q_last:.4g} Å^-1; the r grid allows Q up to "
        f"pi / dr = {math.pi / dr:.4g} Å^-1",
    )
    return texts, widths


def _partial_sq(partials: FramePartials, kernel: np.ndarray
                ) -> dict[tuple[str, str], np.ndarray]:
    pairs = list(partials.g)
    stack = np.stack([partials.g[p] - 1.0 for p in pairs], axis=1)
    values = 1.0 + 4.0 * math.pi * partials.rho0_per_ang3 * (kernel @ stack)
    return {p: values[:, k] for k, p in enumerate(pairs)}


def partial_structure_factors(partials: FramePartials, q_inv_ang, *,
                              r_window: str
                              ) -> dict[tuple[str, str], np.ndarray]:
    """Faber-Ziman S_ab(Q) of one frame by the sine transform of g_ab(r).

    ``r_window`` 'none' or 'Lorch' (no default). Q = 0 is allowed: the
    limit sin(Qr)/(Qr) -> 1 is taken. A Q above pi / dr is refused (there
    the grid sum mirrors a lower Q). :func:`sine_route_notes` states what
    the cut and the grid do to these values.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    r_window = _choice(r_window, "r_window", R_WINDOWS)
    _nyquist_q(float(q[-1]), partials.dr_ang, "q_inv_ang")
    kernel = _sine_kernel(q, partials.r_ang, partials.dr_ang, r_window)
    return {p: _frozen(v) for p, v in _partial_sq(partials, kernel).items()}


# ---------------------------------------------------------------------------
# totals in Q
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class TotalSQ:
    """One frame's total structure factor for one radiation.

    ``s``: Faber-Ziman S(Q); ``f_reduced_inv_ang``: F(Q) = Q [S(Q) - 1] in
    Å^-1; ``f_keen``: F_K(Q) = <f>^2 [S(Q) - 1] in ``weight_unit``;
    ``weights``: w_ab(Q) of each unordered pair (they sum to 1);
    ``f_mean``/``f2_mean``: <f>(Q) and <f^2>(Q) in the factor's own unit
    (fm, electrons, Å).
    """

    radiation: str
    q_inv_ang: np.ndarray
    s: np.ndarray
    f_reduced_inv_ang: np.ndarray
    f_keen: np.ndarray
    weight_unit: str
    weights: Mapping[tuple[str, str], np.ndarray]
    f_mean: np.ndarray
    f2_mean: np.ndarray
    notes: tuple[str, ...] = ()


def _below_bound_note(q: np.ndarray, s: np.ndarray, bound: np.ndarray
                      ) -> list[str]:
    """Where a total S(Q) lies below 1 - <f^2>/<f>^2, which it cannot."""
    below = np.flatnonzero(s < bound)
    if not below.size:
        return []
    depth = bound[below] - s[below]
    worst = int(below[int(np.argmax(depth))])
    return [f"S(Q) lies below 1 - <f^2>/<f>^2, the least a Faber-Ziman total "
            "can take (|sum_j f_j exp(i q.r_j)|^2 >= 0), at "
            f"{below.size} Q point(s) from {float(q[below[0]]):.4g} to "
            f"{float(q[below[-1]]):.4g} Å^-1, by up to {float(depth.max()):.3g} "
            f"(S = {float(s[worst]):.4g} against {float(bound[worst]):.4g} at "
            f"Q = {float(q[worst]):.4g} Å^-1); on the sine route such values "
            "come from the window and the cut at r_max"]


def _total_from_partials(partial_sq, concentrations, q, radiation,
                         lengths, factor_notes=()) -> TotalSQ:
    weights, f_mean, f2_mean = faber_ziman_weights(concentrations, lengths)
    s = np.ones_like(q)
    for pair, w in weights.items():
        s = s + w * (partial_sq[pair] - 1.0)
    unit = _unit_factor(radiation)
    bound = np.broadcast_to(1.0 - f2_mean / (f_mean * f_mean), q.shape)
    notes = (list(factor_notes) + _range_notes(radiation, q, lengths)
             + _below_bound_note(q, s, bound))
    return TotalSQ(
        radiation=radiation, q_inv_ang=q, s=_frozen(s),
        f_reduced_inv_ang=_frozen(q * (s - 1.0)),
        f_keen=_frozen(unit * f_mean * f_mean * (s - 1.0)),
        weight_unit=WEIGHT_UNITS[radiation],
        weights={p: _frozen(np.broadcast_to(w, q.shape))
                 for p, w in weights.items()},
        f_mean=_frozen(np.broadcast_to(f_mean, q.shape)),
        f2_mean=_frozen(np.broadcast_to(f2_mean, q.shape)),
        notes=tuple(notes))


def total_structure_factor(partial_sq: Mapping[tuple[str, str], np.ndarray],
                           concentrations: Mapping[str, float], q_inv_ang,
                           radiation: str, *,
                           neutron_lengths_fm: Mapping[str, float] | None = None,
                           lengths_source: str | None = None) -> TotalSQ:
    """The Faber-Ziman total S(Q), F(Q) and F_K(Q) for one radiation.

    ``partial_sq`` maps every unordered element pair (a <= b) to S_ab on
    ``q_inv_ang``; ``concentrations`` are the atom fractions, and every
    element of a partial has to be among them. Neutron lengths are constant
    in Q (gemmi's natural-abundance lengths unless ``neutron_lengths_fm``
    and ``lengths_source`` give others); X-ray and electron factors are
    taken at each Q. The notes name the factors used, and every Q where the
    total lies below its bound 1 - <f^2>/<f>^2.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    radiation = _radiation(radiation)
    symbols = sorted(concentrations)
    held = {e for pair in partial_sq for e in pair}
    extra = sorted(held - set(symbols))
    if extra:
        raise ValueError(
            f"partial S(Q) of {', '.join(extra)} given, but the "
            f"concentrations hold only {', '.join(symbols)}; a total over "
            "fewer elements than the partials would leave those pairs out")
    missing = [p for p in _unordered_pairs(symbols) if p not in partial_sq]
    if missing:
        raise ValueError(f"partial S(Q) missing for {missing}")
    for pair in _unordered_pairs(symbols):
        if np.shape(partial_sq[pair]) != q.shape:
            raise ValueError(f"S_{_pair_name(pair)} has shape "
                             f"{np.shape(partial_sq[pair])}; the Q grid has "
                             f"{q.size} points")
    lengths = scattering_lengths(symbols, radiation, q,
                                 neutron_lengths_fm=neutron_lengths_fm,
                                 lengths_source=lengths_source)
    given = _neutron_overrides(neutron_lengths_fm, lengths_source, symbols)
    return _total_from_partials(
        partial_sq, concentrations, q, radiation, lengths,
        _factor_notes(symbols, radiation,
                      given if radiation == "neutron" else {},
                      lengths_source))


# ---------------------------------------------------------------------------
# real space
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class RealSpace:
    """One frame's total real-space functions for one radiation (Q = 0 weights).

    ``pdf_g``: G(r) = 4 pi r rho0 [g(r) - 1] in Å^-2, the function
    ``pdf.pair_distribution`` computes; ``keen_g``: G_K(r) in ``weight_unit``;
    ``keen_d``: D(r) = <b>^2 G(r) and ``keen_t``: T(r) = D(r) + 4 pi r rho0
    <b>^2, both in ``weight_unit``/Å^2. ``b_mean_sq``: <b>^2 in
    ``weight_unit``. With ``q_max_inv_ang`` set, G, G_K and D are terminated.
    """

    radiation: str
    r_ang: np.ndarray
    pdf_g: np.ndarray
    keen_g: np.ndarray
    keen_d: np.ndarray
    keen_t: np.ndarray
    weight_unit: str
    b_mean_sq: float
    weights: Mapping[tuple[str, str], float]
    rho0_per_ang3: float
    q_min_inv_ang: float
    q_max_inv_ang: float | None
    q_window: str | None
    sigma_ang: float | None
    notes: tuple[str, ...] = ()


class _LorchOperator:
    """``pdf._lorch_round_trip`` with its two sine matrices built once.

    The same Q grid (``linspace(max(q_min, 1e-6), q_max, n_q)``), the same
    rectangle sums and the same window sin(pi Q / Qmax) / (pi Q / Qmax):
    F = M S_Qr G dr, then G' = (2/pi) S_rQ F dQ. Applied as two
    matrix-vector products per frame (2 n_q n_r terms) rather than as their
    (n_r, n_r) product, which would cost n_r^2 n_q to form: about 500 frames'
    worth at n_q = 4096.
    """

    def __init__(self, r: np.ndarray, q_min: float, q_max: float,
                 n_q: int = 4096):
        q = np.linspace(max(q_min, 1e-6), q_max, n_q)
        dr = float(r[1] - r[0])
        dq = float(q[1] - q[0])
        window_arg = math.pi * q / q_max                  # dimensionless
        sin_qr = np.sin(np.outer(q, r))                   # (n_q, n_r)
        self.forward = (np.sin(window_arg) / window_arg)[:, None] * sin_qr * dr
        self.back = (2.0 / math.pi) * dq * sin_qr.T       # (n_r, n_q)

    def __matmul__(self, g_pdf: np.ndarray) -> np.ndarray:
        return self.back @ (self.forward @ g_pdf)


def _lorch_matrix(r: np.ndarray, q_min: float, q_max: float,
                  n_q: int = 4096) -> _LorchOperator:
    """The Lorch round trip of pdf.py as an operator applied with ``@``."""
    return _LorchOperator(r, q_min, q_max, n_q)


def _termination_args(q_max_inv_ang, q_min_inv_ang, q_window):
    if q_max_inv_ang is None:
        if q_window is not None:
            raise ValueError("q_window was given without q_max_inv_ang")
        q_min = _positive(q_min_inv_ang, "q_min_inv_ang", allow_zero=True)
        if q_min != 0.0:
            raise ValueError(
                f"q_min_inv_ang {q_min:g} was given without q_max_inv_ang; a "
                "termination range needs both ends (and q_window)")
        return None, 0.0, None
    q_max = _positive(q_max_inv_ang, "q_max_inv_ang")
    q_min = _positive(q_min_inv_ang, "q_min_inv_ang", allow_zero=True)
    if q_min >= q_max:
        raise ValueError(f"q_min_inv_ang {q_min} is not below q_max_inv_ang "
                         f"{q_max}")
    if q_window is None:
        raise ValueError(f"q_window is needed with q_max_inv_ang: one of "
                         f"{Q_WINDOWS} (there is no default; the choice "
                         "changes G(r))")
    return q_max, q_min, _choice(q_window, "q_window", Q_WINDOWS)


def _real_space(partials: FramePartials, radiation: str,
                lengths_q0: Mapping[str, float], q_max: float | None,
                q_min: float, q_window: str | None, sigma: float | None,
                lorch: _LorchOperator | None,
                factor_notes: Sequence[str] = ()) -> RealSpace:
    r, dr = partials.r_ang, partials.dr_ang
    weights, f_mean, _ = faber_ziman_weights(partials.concentrations,
                                             lengths_q0)
    f_mean = float(f_mean)
    # pdf.pair_distribution's own order: the weighted deposited counts, then
    # the broadening, then the division by N <b>^2
    weighted = np.zeros_like(r)
    for (a, b), counts in partials.pair_hist.items():
        weighted = weighted + lengths_q0[a] * lengths_q0[b] * counts
    if sigma is not None:
        radial = pdf._broaden(weighted, dr, sigma)
    else:
        radial = weighted / dr
    radial = radial / (partials.n_atoms * f_mean * f_mean)
    rho0 = partials.rho0_per_ang3
    baseline = 4.0 * math.pi * r * rho0
    g_pdf = radial / r - baseline
    notes = list(factor_notes)
    if q_max is not None:
        if q_window == "boxcar":
            g_pdf = pdf.apply_termination(r, g_pdf, q_min, q_max)
        else:
            matrix = lorch if lorch is not None else _lorch_matrix(
                r, q_min, q_max)
            g_pdf = matrix @ g_pdf
        notes.append(
            f"terminated at Qmax = {q_max:g} Å^-1"
            + (f" from Qmin = {q_min:g} Å^-1" if q_min else "")
            + f" with a {q_window} window, as pdf.py applies it; the model's "
            f"G(r) stops at r_max = {r[-1]:.6g} Å, so within a few pi/Qmax "
            f"({math.pi / q_max:.3g} Å) of r_max the convolution lacks the "
            "pairs beyond it")
    b2 = _unit_factor(radiation) * f_mean * f_mean
    keen_d = b2 * g_pdf
    keen_t = keen_d + b2 * baseline
    keen_g = keen_d / baseline
    if radiation != "neutron":
        notes.append(f"{radiation} weights at Q = 0, as pdf.py takes them; a "
                     f"measured {radiation} G(r) is the transform of S(Q) "
                     "normalised by <f(Q)>^2 at each Q (real_space_from_sq), "
                     "which differs from this one where the weights change "
                     "with Q")
    if sigma is not None:
        notes.append(f"broadened by one Gaussian of sigma = {sigma:g} Å "
                     "(pdf._broaden), as pdf.pair_distribution broadens")
    return RealSpace(
        radiation=radiation, r_ang=r, pdf_g=_frozen(g_pdf),
        keen_g=_frozen(keen_g), keen_d=_frozen(keen_d),
        keen_t=_frozen(keen_t), weight_unit=WEIGHT_UNITS[radiation],
        b_mean_sq=b2, weights={p: float(w) for p, w in weights.items()},
        rho0_per_ang3=rho0, q_min_inv_ang=q_min, q_max_inv_ang=q_max,
        q_window=q_window, sigma_ang=sigma, notes=tuple(notes))


def real_space_from_partials(partials: FramePartials, radiation: str, *,
                             q_max_inv_ang: float | None = None,
                             q_min_inv_ang: float = 0.0,
                             q_window: str | None = None,
                             sigma_ang: float | None = None,
                             neutron_lengths_fm: Mapping[str, float] | None = None,
                             lengths_source: str | None = None) -> RealSpace:
    """G(r), G_K(r), D(r) and T(r) of one frame for one radiation.

    Untruncated unless ``q_max_inv_ang`` is given, and then ``q_window``
    ('boxcar' or 'Lorch') is required; a Qmax above pi / dr is refused (the
    sampled termination kernel aliases there: at 2 pi / dr it is twice the
    identity). ``sigma_ang``: an optional Gaussian broadening, as pdf.py
    applies to a crystal; an MD model carries its own thermal spread, so
    None is the plain model. X-ray and electron weights are taken at Q = 0,
    as in pdf.py. ``neutron_lengths_fm`` / ``lengths_source``: as in
    :func:`scattering_lengths`.
    """
    radiation = _radiation(radiation)
    q_max, q_min, q_window = _termination_args(q_max_inv_ang, q_min_inv_ang,
                                               q_window)
    if q_max is not None:
        _nyquist_q(q_max, partials.dr_ang, "q_max_inv_ang")
    sigma = None if sigma_ang is None else _positive(sigma_ang, "sigma_ang")
    lengths = scattering_lengths(partials.elements, radiation,
                                 neutron_lengths_fm=neutron_lengths_fm,
                                 lengths_source=lengths_source)
    given = _neutron_overrides(neutron_lengths_fm, lengths_source,
                               partials.elements)
    notes = _factor_notes(partials.elements, radiation,
                          given if radiation == "neutron" else {},
                          lengths_source)
    return _real_space(partials, radiation, lengths, q_max, q_min, q_window,
                       sigma, None, notes)


def _sq_to_g_matrix(q: np.ndarray, r: np.ndarray, q_min: float,
                    q_max: float, window: str
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The matrix A with G(r) = A @ (S(Q) - 1) over the Q points inside
    [Qmin, Qmax], and the mask of those points.

    A[r, k] = (2/pi) sin(Q_k r) Q_k M(Q_k) w_k, w_k the local spacing
    (:func:`_q_weights`). The data have to cover [Qmin, Qmax] to within half
    a step at each end, and r may not pass pi / (the largest step), where the
    Q sum mirrors a smaller r.
    """
    if q.size < 2:
        raise ValueError("q_inv_ang: at least two points are needed")
    first_step, last_step = float(q[1] - q[0]), float(q[-1] - q[-2])
    if q_min < float(q[0]) - 0.5 * first_step:
        raise ValueError(
            f"q_min_inv_ang {q_min:g} lies below the first Q of the data, "
            f"{float(q[0]):.6g} Å^-1; the transform would start there "
            f"without saying so. q_min_inv_ang >= {float(q[0]):.6g} states "
            "the range the data cover")
    if q_max > float(q[-1]) + 0.5 * last_step:
        raise ValueError(
            f"q_max_inv_ang {q_max:g} lies beyond the last Q of the data, "
            f"{float(q[-1]):.6g} Å^-1; the transform would stop there while "
            "its window was built for the Qmax asked for. q_max_inv_ang <= "
            f"{float(q[-1]):.6g} states the range the data cover")
    inside = (q >= q_min) & (q <= q_max)
    if inside.sum() < 2:
        raise ValueError(f"fewer than two Q points lie in [{q_min:g}, "
                         f"{q_max:g}] Å^-1")
    qq = q[inside]
    largest = float(np.diff(qq).max())
    if float(r[-1]) > math.pi / largest * (1.0 + 1e-12):
        raise ValueError(
            f"r_ang reaches {float(r[-1]):.6g} Å, beyond pi / dQ = "
            f"{math.pi / largest:.6g} Å for the largest Q step "
            f"{largest:.6g} Å^-1: the sum over the Q points is periodic in "
            "r, so beyond it G(r) mirrors a smaller r. A finer Q grid or a "
            "shorter r grid is needed")
    size = r.size * qq.size
    if size > KERNEL_VALUES_LIMIT:
        raise ValueError(
            f"the Q -> r matrix would hold {r.size} x {qq.size} = {size} "
            f"values, above KERNEL_VALUES_LIMIT ({KERNEL_VALUES_LIMIT}, a "
            "memory choice); a coarser Q grid or a shorter r grid needs fewer")
    factor = qq * _q_weights(qq)
    if window == "Lorch":
        factor = factor * np.sinc(qq / q_max)
    return (2.0 / math.pi) * np.sin(np.outer(r, qq)) * factor[None, :], inside


def real_space_from_sq(q_inv_ang, s, r_ang, *, q_min_inv_ang: float,
                       q_max_inv_ang: float, q_window: str) -> np.ndarray:
    """G(r) = (2/pi) Int_Qmin^Qmax Q [S(Q) - 1] M(Q) sin(Qr) dQ.

    A sum over the Q points inside [Qmin, Qmax], each weighted by its local
    spacing, which on a uniform grid is ``pdf.inverse_transform``'s
    rectangle rule (a test holds the two equal) and on a grid that is
    uniform only to its printed digits, or not uniform, still a quadrature
    of the integral. M(Q) = 1 ('boxcar') or sin(pi Q/Qmax)/(pi Q/Qmax)
    ('Lorch'). Refused: a Qmin or Qmax outside the data (by more than half a
    step), a NaN or infinite S inside the range, and r beyond pi / (the
    largest Q step). For X-rays this is the transform of an S(Q) normalised
    by <f(Q)>^2 at every Q, which differs from the Q = 0 weighted G(r) of
    :func:`real_space_from_partials`.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    values = np.asarray(s, dtype=np.float64)
    if values.shape != q.shape:
        raise ValueError(f"s has shape {values.shape}; q_inv_ang has "
                         f"{q.shape}")
    r = _axis(r_ang, "r_ang")
    if q_max_inv_ang is None:
        raise ValueError("q_max_inv_ang is needed: the transform runs over "
                         "[q_min_inv_ang, q_max_inv_ang]")
    q_max, q_min, window = _termination_args(q_max_inv_ang, q_min_inv_ang,
                                             q_window)
    matrix, inside = _sq_to_g_matrix(q, r, q_min, q_max, window)
    used = values[inside]
    nonfinite = np.flatnonzero(~np.isfinite(used))
    if nonfinite.size:
        where = q[inside][nonfinite]
        raise ValueError(
            f"s: {nonfinite.size} value(s) inside [{q_min:g}, {q_max:g}] Å^-1 are "
            f"NaN or infinite (first at Q = {float(where[0]):.6g} Å^-1); "
            "G(r) would be NaN at every r")
    return _frozen(matrix @ (used - 1.0))


# ---------------------------------------------------------------------------
# Bhatia-Thornton
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BhatiaThornton:
    """S_NN, S_NC, S_CC of a binary model; A is the first element."""

    elements: tuple[str, str]
    c_a: float
    c_b: float
    s_nn: np.ndarray
    s_nc: np.ndarray
    s_cc: np.ndarray
    notes: tuple[str, ...] = ()


def bhatia_thornton(partial_sq: Mapping[tuple[str, str], np.ndarray],
                    concentrations: Mapping[str, float]) -> BhatiaThornton:
    """The Bhatia-Thornton functions from the Faber-Ziman partials.

    Defined for two elements only; ValueError otherwise. A and B are the two
    elements in alphabetical order, and the concentration mode is
    C(q) = c_B rho_A(q) - c_A rho_B(q), which fixes the sign of S_NC.
    """
    symbols = sorted(concentrations)
    if len(symbols) != 2:
        raise ValueError(f"the Bhatia-Thornton functions are defined for a "
                         f"binary model; this one has {len(symbols)} elements "
                         f"({', '.join(symbols)})")
    check_fractions([concentrations[s] for s in symbols], "concentrations")
    a, b = symbols
    for pair in ((a, a), (a, b), (b, b)):
        if pair not in partial_sq:
            raise ValueError(f"partial S(Q) missing for {_pair_name(pair)}")
    s_aa = np.asarray(partial_sq[(a, a)], dtype=np.float64)
    s_ab = np.asarray(partial_sq[(a, b)], dtype=np.float64)
    s_bb = np.asarray(partial_sq[(b, b)], dtype=np.float64)
    ca, cb = float(concentrations[a]), float(concentrations[b])
    s_nn = ca * ca * s_aa + cb * cb * s_bb + 2.0 * ca * cb * s_ab
    s_nc = ca * cb * (ca * (s_aa - s_ab) - cb * (s_bb - s_ab))
    s_cc = ca * cb * (1.0 + ca * cb * (s_aa + s_bb - 2.0 * s_ab))
    return BhatiaThornton(
        elements=(a, b), c_a=ca, c_b=cb, s_nn=_frozen(s_nn),
        s_nc=_frozen(s_nc), s_cc=_frozen(s_cc),
        notes=(f"A = {a}, B = {b}; C(q) = c_B rho_A(q) - c_A rho_B(q); "
               f"S_CC tends to c_A c_B = {ca * cb:.6g} at high Q",))


# ---------------------------------------------------------------------------
# the second route: S(q) from the positions
# ---------------------------------------------------------------------------

def _reciprocal_basis(frame: Frame) -> np.ndarray:
    """Rows 2 pi a*, 2 pi b*, 2 pi c*: q = (h, k, l) @ this, q . r = 2 pi hkl . f."""
    return 2.0 * math.pi * np.linalg.inv(frame.box_ang).T


def density_modes(frame: Frame, hkl) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """rho_e(q) = sum_{j of element e} exp(i q . r_j) at lattice vectors hkl.

    ``hkl``: (M, 3) integers, indices on the reciprocal lattice of the box.
    Returns the (M, 3) vectors q in Å^-1 and, per element, the (M,) complex
    sums. Positions are taken from the fractions, so q . r_j =
    2 pi (h f1 + k f2 + l f3) exactly; the box origin only multiplies every
    sum by one phase, which no product rho_a rho_b* sees. The direct sum,
    kept as the reference for :func:`sq_from_positions`.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    indices = np.asarray(hkl)
    if indices.ndim != 2 or indices.shape[1] != 3 or \
            indices.dtype.kind not in "iu":
        raise ValueError("hkl needs an (M, 3) array of integers")
    vectors = indices.astype(np.float64) @ _reciprocal_basis(frame)
    out: dict[str, np.ndarray] = {}
    for symbol in frame.species:
        frac = frame.frac[frame.elements == symbol]
        phase = 2.0 * math.pi * (frac @ indices.T.astype(np.float64))
        out[symbol] = np.exp(1j * phase).sum(axis=0)
    return vectors, out


@dataclass(frozen=True, eq=False)
class DirectSQ:
    """S(q) of one frame from the positions, averaged over shells of |q|.

    ``n_vectors[k]`` lattice vectors fall in shell k (q and -q both counted;
    they give the same value). ``q_mean_inv_ang``: their mean |q|, NaN for an
    empty shell. ``partial_s``: Faber-Ziman S_ab per shell, ``partial_std``
    their spread over the vectors of the shell; ``total_s`` and
    ``total_std`` per radiation asked for. Partials of a binary model also
    give Bhatia-Thornton through :func:`bhatia_thornton`; ``bt_modes`` holds
    S_NN, S_NC and S_CC computed directly from the number and concentration
    modes (binary models only).
    """

    q_edges_inv_ang: np.ndarray
    q_mean_inv_ang: np.ndarray
    n_vectors: np.ndarray
    elements: tuple[str, ...]
    concentrations: Mapping[str, float]
    partial_s: Mapping[tuple[str, str], np.ndarray]
    partial_std: Mapping[tuple[str, str], np.ndarray]
    total_s: Mapping[str, np.ndarray]
    total_std: Mapping[str, np.ndarray]
    bt_modes: Mapping[str, np.ndarray] | None
    notes: tuple[str, ...] = ()


def sq_from_positions(frame: Frame, q_edges_inv_ang, *,
                      radiations: Sequence[str] = (),
                      neutron_lengths_fm: Mapping[str, float] | None = None,
                      lengths_source: str | None = None) -> DirectSQ:
    """Faber-Ziman S(q) at every reciprocal-lattice vector, averaged in shells.

    ``q_edges_inv_ang``: shell edges in Å^-1 (strictly increasing, >= 0); a
    vector belongs to shell k when edges[k] <= |q| < edges[k + 1]; q = 0 is
    never included. ``radiations``: the totals to form (none, any of
    :data:`RADIATIONS`), each with its factors at the vector's own |q|;
    ``neutron_lengths_fm`` / ``lengths_source`` as in
    :func:`scattering_lengths`.

    Cost: the sum runs over (H0 + 1) (2 H1 + 1) (2 H2 + 1) N terms with
    H_i = floor(q_max |a_i| / 2 pi), so it grows as q_max^3 V N.
    """
    if not isinstance(frame, Frame):
        raise ValueError(f"a Frame is needed, not {type(frame).__name__}")
    edges = _axis(q_edges_inv_ang, "q_edges_inv_ang")
    if edges.size < 2:
        raise ValueError("q_edges_inv_ang: at least two edges are needed")
    radiations = _radiation_list(radiations)
    overrides = dict(neutron_lengths_fm=neutron_lengths_fm,
                     lengths_source=lengths_source)
    n = frame.n_atoms
    elements = frame.species
    counts = frame.composition
    conc = {s: counts[s] / n for s in elements}
    groups = [np.flatnonzero(frame.elements == s) for s in elements]
    basis = _reciprocal_basis(frame)
    q_hi = float(edges[-1])
    lengths = np.linalg.norm(frame.box_ang, axis=1)
    reach = np.floor(q_hi * lengths / (2.0 * math.pi) + 1e-9).astype(int)
    for axis in (1, 2):
        size = n * (2 * int(reach[axis]) + 1)
        if size > MODES_VALUES_LIMIT:
            raise ValueError(
                f"the reciprocal-lattice route would hold {size} complex "
                f"values per axis table ({n} atoms x {2 * reach[axis] + 1} "
                f"indices), above MODES_VALUES_LIMIT ({MODES_VALUES_LIMIT}, "
                "a memory choice); a smaller q_max needs fewer")
    frac = frame.frac
    k_idx = np.arange(-reach[1], reach[1] + 1)
    l_idx = np.arange(-reach[2], reach[2] + 1)
    e1 = np.exp(2j * math.pi * np.outer(frac[:, 1], k_idx))   # (N, 2H1+1)
    e2 = np.exp(2j * math.pi * np.outer(frac[:, 2], l_idx))   # (N, 2H2+1)
    kk, ll = np.meshgrid(k_idx, l_idx, indexing="ij")
    plane = (kk[..., None] * basis[1] + ll[..., None] * basis[2])

    pairs = _unordered_pairs(list(elements))
    n_shells = edges.size - 1
    w_sum = np.zeros(n_shells)
    q_sum = np.zeros(n_shells)
    p_sum = {p: np.zeros(n_shells) for p in pairs}
    p_sq = {p: np.zeros(n_shells) for p in pairs}
    t_sum = {radiation: np.zeros(n_shells) for radiation in radiations}
    t_sq = {radiation: np.zeros(n_shells) for radiation in radiations}
    binary = len(elements) == 2
    bt_sum = {k: np.zeros(n_shells) for k in ("S_NN", "S_NC", "S_CC")} \
        if binary else None
    # the factors at Q = 0 decide nothing here; they are checked up front so
    # a missing table is refused before the sums run
    given = _neutron_overrides(neutron_lengths_fm, lengths_source, elements)
    for radiation in radiations:
        scattering_lengths(elements, radiation, **overrides)

    for h in range(0, int(reach[0]) + 1):
        q_vec = plane + h * basis[0]
        q_norm = np.sqrt((q_vec * q_vec).sum(axis=-1))
        chosen = (q_norm >= edges[0]) & (q_norm < q_hi) & (q_norm > 0.0)
        if not chosen.any():
            continue
        qn = q_norm[chosen]
        shell = np.searchsorted(edges, qn, side="right") - 1
        weight = np.full(qn.size, 2.0 if h > 0 else 1.0)
        a_h = np.exp(2j * math.pi * h * frac[:, 0])[:, None] * e1
        rho = [(a_h[idx].T @ e2[idx])[chosen] for idx in groups]

        def add(total, squares, values):
            total += np.bincount(shell, weights=weight * values,
                                 minlength=n_shells)
            squares += np.bincount(shell, weights=weight * values * values,
                                   minlength=n_shells)

        w_sum += np.bincount(shell, weights=weight, minlength=n_shells)
        q_sum += np.bincount(shell, weights=weight * qn, minlength=n_shells)
        for ia, a in enumerate(elements):
            for ib in range(ia, len(elements)):
                b = elements[ib]
                cross = (rho[ia] * np.conj(rho[ib])).real / n
                if ia == ib:
                    cross = cross - conc[a]
                add(p_sum[(a, b)], p_sq[(a, b)],
                    1.0 + cross / (conc[a] * conc[b]))
        for radiation in radiations:
            f = scattering_lengths(elements, radiation, qn, **overrides)
            amplitude = sum(f[s] * rho[k] for k, s in enumerate(elements))
            f_mean = sum(conc[s] * f[s] for s in elements)
            f2_mean = sum(conc[s] * f[s] * f[s] for s in elements)
            intensity = (amplitude * np.conj(amplitude)).real / n
            add(t_sum[radiation], t_sq[radiation],
                1.0 + (intensity - f2_mean) / (f_mean ** 2))
        if binary:
            ca, cb = conc[elements[0]], conc[elements[1]]
            number = rho[0] + rho[1]
            concentration = cb * rho[0] - ca * rho[1]
            for key, values in (
                    ("S_NN", (number * np.conj(number)).real / n),
                    ("S_NC", (number * np.conj(concentration)).real / n),
                    ("S_CC", (concentration * np.conj(concentration)).real
                     / n)):
                bt_sum[key] += np.bincount(shell, weights=weight * values,
                                           minlength=n_shells)

    with np.errstate(invalid="ignore", divide="ignore"):
        def mean(total):
            return np.where(w_sum > 0, total / w_sum, np.nan)

        def spread(total, squares):
            m = mean(total)
            return np.sqrt(np.maximum(mean(squares) - m * m, 0.0))

        q_mean = mean(q_sum)
        partial_s = {p: _frozen(mean(p_sum[p])) for p in pairs}
        partial_std = {p: _frozen(spread(p_sum[p], p_sq[p])) for p in pairs}
        total_s = {name: _frozen(mean(t_sum[name])) for name in radiations}
        total_std = {name: _frozen(spread(t_sum[name], t_sq[name]))
                     for name in radiations}
        bt = None if bt_sum is None else {k: _frozen(mean(v))
                                          for k, v in bt_sum.items()}
    empty = int((w_sum == 0).sum())
    notes = [f"Faber-Ziman S(q) at the reciprocal-lattice vectors of the box "
             f"with {edges[0]:g} <= |q| < {q_hi:g} Å^-1, {int(w_sum.sum())} "
             f"vectors in {n_shells} shells (q and -q both counted, with the "
             "same value); std is the spread over the vectors of a shell"]
    if empty:
        notes.append(f"{empty} of {n_shells} shells hold no lattice vector "
                     "(the spacing is 2 pi / L); their values are NaN")
    for radiation in radiations:
        notes.extend(_factor_notes(elements, radiation,
                                   given if radiation == "neutron" else {},
                                   lengths_source))
        notes.extend(_range_notes(radiation, edges, scattering_lengths(
            elements, radiation, edges, **overrides)))
    return DirectSQ(
        q_edges_inv_ang=edges, q_mean_inv_ang=_frozen(q_mean),
        n_vectors=np.round(w_sum).astype(np.int64), elements=elements,
        concentrations=conc, partial_s=partial_s, partial_std=partial_std,
        total_s=total_s, total_std=total_std, bt_modes=bt,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# the first sharp diffraction peak
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FSDP:
    """Position, height and width of the highest point of S(Q) in a window."""

    q_peak_inv_ang: float
    height: float
    fwhm_inv_ang: float
    q_left_inv_ang: float
    q_right_inv_ang: float
    baseline: str
    baseline_at_peak: float
    window_inv_ang: tuple[float, float]
    method: str
    notes: tuple[str, ...] = ()

    @property
    def repeat_distance_ang(self) -> float:
        """2 pi / Q_FSDP."""
        return 2.0 * math.pi / self.q_peak_inv_ang

    @property
    def coherence_length_ang(self) -> float:
        """2 pi / FWHM."""
        return 2.0 * math.pi / self.fwhm_inv_ang


_FSDP_METHOD = (
    "position and height: vertex of the parabola through the highest grid "
    "point in the window and its two neighbours; width: full width at half "
    "height above the baseline, crossings interpolated linearly between grid "
    "points, searched inside the window only")


def _crossing(q, d, start, step, stop):
    """First Q where d changes sign walking from ``start`` by ``step``."""
    k = start
    while k != stop:
        nxt = k + step
        if d[nxt] <= 0.0:
            # linear interpolation between k (d > 0) and nxt (d <= 0)
            t = d[k] / (d[k] - d[nxt])
            return float(q[k] + t * (q[nxt] - q[k]))
        k = nxt
    return float("nan")


def fsdp(q_inv_ang, s, *, q_window_inv_ang: tuple[float, float],
         baseline: str) -> FSDP:
    """The first sharp diffraction peak of S(Q), measured in a stated window.

    ``q_window_inv_ang``: (low, high) in Å^-1, where to look (no default).
    ``baseline``: 'zero' (half height = S_peak / 2) or 'minima' (the line
    through the lowest point on each side of the peak within the window).
    A maximum at the window's edge, or a half-height crossing not found in
    the window, gives NaN values and a note; so does a peak that rises 0
    or less above its baseline (a negative maximum over 'zero'), which has
    no half height.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    values = np.asarray(s, dtype=np.float64)
    if values.shape != q.shape:
        raise ValueError(f"s has shape {values.shape}; q_inv_ang has {q.shape}")
    low, high = _range_pair(q_window_inv_ang, "q_window_inv_ang")
    baseline = _choice(baseline, "baseline", FSDP_BASELINES)
    inside = np.flatnonzero((q >= low) & (q <= high))
    nan = float("nan")

    def empty(note: str) -> FSDP:
        return FSDP(nan, nan, nan, nan, nan, baseline, nan, (low, high),
                    _FSDP_METHOD, (note,))

    if inside.size < 3:
        return empty(f"fewer than three Q points in the window [{low:g}, "
                     f"{high:g}] Å^-1; no peak measured")
    if not np.isfinite(values[inside]).all():
        return empty("a value of S(Q) in the window is NaN; no peak measured")
    first, last = int(inside[0]), int(inside[-1])
    top = first + int(np.argmax(values[first:last + 1]))
    if top in (first, last):
        return empty(f"the largest S(Q) in the window lies at its edge "
                     f"(Q = {q[top]:.4g} Å^-1); no peak inside the window")
    x3, y3 = q[top - 1:top + 2], values[top - 1:top + 2]
    curvature, slope, offset = np.polyfit(x3 - q[top], y3, 2)
    if curvature < 0.0:
        shift = -slope / (2.0 * curvature)
        q_peak = float(q[top] + shift)
        height = float(offset + slope * shift + curvature * shift * shift)
    else:                                   # three collinear points
        q_peak, height = float(q[top]), float(values[top])
    notes = []
    if baseline == "zero":
        base = np.zeros_like(values)
    else:
        left_min = first + int(np.argmin(values[first:top + 1]))
        right_min = top + int(np.argmin(values[top:last + 1]))
        q_l, q_r = q[left_min], q[right_min]
        slope_b = (values[right_min] - values[left_min]) / (q_r - q_l)
        base = values[left_min] + slope_b * (q - q_l)
        notes.append(f"baseline through S({q_l:.4g}) = {values[left_min]:.6g} "
                     f"and S({q_r:.4g}) = {values[right_min]:.6g}")
    base_peak = float(np.interp(q_peak, q, base))
    rise = height - base_peak
    d = values - base - 0.5 * rise
    if not rise > 0.0:
        # a negative 'peak' over the 'zero' baseline (a partial S_ab, S_NC)
        left = right = nan
        notes.append(f"the peak rises {rise:.4g} above the '{baseline}' "
                     "baseline, 0 or less, so it has no half height; the "
                     "width is not measured")
    elif not d[top] > 0.0:
        left = right = nan
        notes.append("the highest grid point lies at or below half the "
                     "peak's height above the baseline at its vertex, so no "
                     "half-height crossing is bracketed; the width is not "
                     "measured")
    else:
        left = _crossing(q, d, top, -1, first)
        right = _crossing(q, d, top, +1, last)
        if math.isnan(left) or math.isnan(right):
            notes.append("a half-height crossing lies outside the window; "
                         "the width is not measured")
    return FSDP(q_peak_inv_ang=q_peak, height=height,
                fwhm_inv_ang=right - left, q_left_inv_ang=left,
                q_right_inv_ang=right, baseline=baseline,
                baseline_at_peak=base_peak, window_inv_ang=(low, high),
                method=_FSDP_METHOD, notes=tuple(notes))


# ---------------------------------------------------------------------------
# comparison with a measured curve
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Measured:
    """A measured curve as read: its axis (named with its unit), its values
    and what the reader set aside or changed."""

    path: str
    axis_name: str                     # 'q_inv_ang' or 'r_ang'
    axis_values: np.ndarray
    values: np.ndarray
    notes: tuple[str, ...] = ()


# A token written with a decimal comma: '0,50', '-1,2e-3', ',5'.
_DECIMAL_COMMA = re.compile(r"^[+-]?\d*,\d+([eE][+-]?\d+)?$")
# Column separators: whitespace, commas and semicolons.
_SEPARATORS = re.compile(r"[\s,;]+")
_COMMENT_START = "#!';*/"          # diffraction.read_pattern's comment marks


def _measured_text(target: Path) -> tuple[str, list[str]]:
    """The file's text, its byte-order mark removed, and what was done."""
    try:
        raw = target.read_bytes()
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise ValueError(f"{target}: the file could not be read ({reason}); "
                         "read_measured needs a text file of two columns") \
            from None
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return (raw.decode("utf-16", errors="replace"),
                ["the file is UTF-16 (it begins with a UTF-16 byte-order "
                 "mark) and was read as such"])
    notes = []
    if raw.startswith(b"\xef\xbb\xbf"):
        notes.append("the file begins with a UTF-8 byte-order mark (as "
                     "Excel's 'CSV UTF-8' writes), removed before reading")
    return raw.decode("utf-8-sig", errors="replace"), notes


def read_measured(path, *, axis_name: str) -> Measured:
    """A two-column measured curve: axis, then value.

    ``axis_name``: what the first column holds, 'q_inv_ang' (Q in Å^-1) or
    'r_ang' (r in Å); there is no default, because the file cannot say. The
    rules are ``diffraction.read_pattern``'s: lines starting with
    ``# ! ' ; * /`` are comments, columns are separated by spaces, tabs or
    commas (and here also semicolons), lines with fewer than two numbers or a
    non-finite one are set aside and counted in a note, and the points are
    sorted by the axis. In addition a UTF-8 byte-order mark is removed, a
    UTF-16 file is read, and a file written with decimal commas
    ('0,50<tab>1,234') is refused with ValueError naming the line, since the
    comma rule would read it as other numbers. A file that cannot be read,
    or holds fewer than two points, is refused with ValueError naming it.
    """
    axis_name = _choice(axis_name, "axis_name", MEASURED_AXES)
    target = Path(path)
    text, notes = _measured_text(target)
    axis, values = [], []
    candidates = 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped[0] in _COMMENT_START:
            continue
        candidates += 1
        tokens = [t for t in re.split(r"[\s;]+", stripped) if t]
        if len(tokens) >= 2 and any(_DECIMAL_COMMA.match(t) for t in tokens):
            raise ValueError(
                f"{target.name}, line {number} ({stripped[:40]!r}): a number "
                "is written with a decimal comma. read_measured reads '.' as "
                "the decimal mark and ',' as a column separator, so this line "
                "would give other numbers; the file needs '.' decimals")
        parts = [t for t in _SEPARATORS.split(stripped) if t]
        if len(parts) < 2:
            continue
        try:
            first, second = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(first) and math.isfinite(second)):
            continue
        axis.append(first)
        values.append(second)
    if len(axis) < 2:
        raise ValueError(
            f"{target.name}: fewer than two lines with two numbers were found. "
            "read_measured expects two columns, the axis (Q in Å^-1 or r in "
            "Å) and the value, separated by spaces, tabs, commas or "
            "semicolons, with '.' as the decimal mark")
    order = np.argsort(np.asarray(axis), kind="stable")
    axis_values = np.asarray(axis)[order]
    notes.insert(0, f"{axis_values.size} points read from {target.resolve()} "
                    f"(axis {axis_name}), sorted by the axis")
    if candidates > axis_values.size:
        notes.append(f"{candidates - axis_values.size} non-comment line(s) "
                     "held no two finite numbers and were set aside (a "
                     "header line counts here)")
    return Measured(path=str(target.resolve()), axis_name=axis_name,
                    axis_values=_frozen(axis_values),
                    values=_frozen(np.asarray(values)[order]),
                    notes=tuple(notes))


@dataclass(frozen=True, eq=False)
class Comparison:
    """A model curve against a measured one: one scale factor and R_chi.

    ``axis_values``: the measured points used, on the axis ``axis_name``
    ('q_inv_ang' or 'r_ang'); ``model``: the scaled model interpolated onto
    them. ``model_step`` / ``measured_step``: the largest model spacing and
    the smallest measured spacing inside the range, so that two R_chi values
    from model grids of different steps can be told apart (the linear
    interpolation of the model enters R_chi). ``measured_source`` and
    ``model_notes``: what the caller said the two curves are.
    """

    quantity: str
    axis_name: str
    axis_values: np.ndarray
    measured: np.ndarray
    model: np.ndarray                  # scaled, on the measured axis values
    difference: np.ndarray             # measured - model
    scale: float
    scale_source: str                  # 'least squares' | 'given'
    r_chi: float
    n_points: int
    n_outside: int
    axis_range: tuple[float, float]
    model_step: float
    measured_step: float
    measured_source: str | None = None
    model_notes: tuple[str, ...] = ()
    definition: str = R_CHI_DEFINITION
    notes: tuple[str, ...] = ()


def compare_with_measured(model_axis, model_values, measured_axis,
                          measured_values, *, axis_name: str, quantity: str,
                          axis_range: tuple[float, float] | None = None,
                          scale: float | None = None,
                          measured_source: str | None = None,
                          model_notes: Sequence[str] = ()) -> Comparison:
    """R_chi of a model curve against a measured one (definition in the result).

    ``axis_name``: 'q_inv_ang' or 'r_ang', the axis both curves are on (no
    default; the two have to be in the same unit, which is not checked
    beyond a note giving both spans). ``quantity`` names what both curves
    are (e.g. 'S(Q) neutron', 'G(r) X-ray'); it is stored, not interpreted.
    Only measured points inside the model's axis range and inside
    ``axis_range`` are used; the others are counted in ``n_outside``.
    ``scale`` None fits the one factor by least squares on the points used;
    a number is applied as given. ``measured_source`` (such as
    ``Measured.path``) and ``model_notes`` (such as the notes of the model
    Series) are stored with the result.
    """
    axis_name = _choice(axis_name, "axis_name", MEASURED_AXES)
    if not isinstance(quantity, str) or not quantity.strip():
        raise ValueError("quantity needs a name, such as 'S(Q) neutron'")
    if measured_source is not None and not isinstance(measured_source, str):
        raise ValueError("measured_source needs a text, such as the file path")
    if isinstance(model_notes, (str, bytes)):
        raise ValueError("model_notes needs a sequence of texts")
    model_notes = tuple(str(note) for note in model_notes)
    mx = _axis(model_axis, "model_axis", nonnegative=False)
    my = np.asarray(model_values, dtype=np.float64)
    if my.shape != mx.shape or not np.isfinite(my).all():
        raise ValueError("model_values needs one finite value per model_axis "
                         "point")
    try:
        ex = np.asarray(measured_axis, dtype=np.float64)
        ey = np.asarray(measured_values, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError("measured_axis and measured_values need numbers") \
            from None
    if ex.ndim != 1 or ey.shape != ex.shape or ex.size == 0:
        raise ValueError("measured_axis and measured_values need the same "
                         "1-D shape")
    if not (np.isfinite(ex).all() and np.isfinite(ey).all()):
        raise ValueError("a measured value is NaN or infinite")
    lo, hi = float(mx[0]), float(mx[-1])
    if axis_range is not None:
        a, b = _range_pair(axis_range, "axis_range")
        if a >= hi or b <= lo:
            raise ValueError(
                f"axis_range ({a:g}, {b:g}) does not overlap the model's "
                f"axis, which spans {lo:.6g} .. {hi:.6g} ({axis_name})")
        lo, hi = max(lo, a), min(hi, b)
    used = (ex >= lo) & (ex <= hi)
    n_used = int(used.sum())
    if n_used < 2:
        raise ValueError(f"fewer than two measured points lie in "
                         f"[{lo:g}, {hi:g}], where the model is defined "
                         f"(the measured axis spans {float(ex.min()):.6g} .. "
                         f"{float(ex.max()):.6g})")
    order = np.argsort(ex[used], kind="stable")
    axis_used, y_meas = ex[used][order], ey[used][order]
    y_model = np.interp(axis_used, mx, my)
    if scale is None:
        denominator = float(np.dot(y_model, y_model))
        if denominator == 0.0:
            raise ValueError("the model is 0 at every point used, so no scale "
                             "factor can be fitted")
        factor = float(np.dot(y_meas, y_model)) / denominator
        source = "least squares"
    else:
        factor = _number(scale, "scale")
        source = "given"
    scaled = factor * y_model
    norm = float(np.dot(y_meas, y_meas))
    if norm == 0.0:
        raise ValueError("the measured curve is 0 at every point used; R_chi "
                         "divides by its sum of squares")
    difference = y_meas - scaled
    r_chi = math.sqrt(float(np.dot(difference, difference)) / norm)
    spans = (mx[1:] >= lo) & (mx[:-1] <= hi)
    model_step = float(np.diff(mx)[spans].max()) if spans.any() else \
        float("nan")
    gaps = np.diff(axis_used)
    gaps = gaps[gaps > 0.0]
    measured_step = float(gaps.min()) if gaps.size else float("nan")
    notes = [f"{n_used} measured points in [{lo:g}, {hi:g}] ({axis_name}) "
             "used; the model interpolated linearly onto them",
             f"the measured axis spans {float(ex.min()):.6g} .. "
             f"{float(ex.max()):.6g}, the model's {float(mx[0]):.6g} .. "
             f"{float(mx[-1]):.6g}",
             f"scale factor {factor:.6g} ({source})",
             "R_chi depends on which function is compared and over what "
             "range; both are stored with it"]
    # 1e-9 relative: two grids of the same step differ by rounding only
    if model_step > measured_step * (1.0 + 1e-9):
        notes.append(f"the model grid's largest step inside the range "
                     f"({model_step:.4g}) exceeds the smallest measured step "
                     f"({measured_step:.4g}), so the model's linear "
                     "interpolation between its points enters R_chi")
    n_out = ex.size - n_used
    if n_out:
        notes.append(f"{n_out} of {ex.size} measured points lie outside that "
                     "range and are not in R_chi")
    return Comparison(quantity=quantity, axis_name=axis_name,
                      axis_values=_frozen(axis_used), measured=_frozen(y_meas),
                      model=_frozen(scaled), difference=_frozen(difference),
                      scale=factor, scale_source=source, r_chi=r_chi,
                      n_points=n_used, n_outside=n_out, axis_range=(lo, hi),
                      model_step=model_step, measured_step=measured_step,
                      measured_source=measured_source,
                      model_notes=model_notes, notes=tuple(notes))


# ---------------------------------------------------------------------------
# a trajectory, frame by frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class TotalScattering:
    """Frame-averaged total functions for one radiation (md_stats.Series).

    ``pdf_g`` (and G_K, D, T) use the weights at Q = 0, as pdf.py does.
    ``pdf_g_from_sq``: G(r) transformed from each frame's total S(Q),
    normalised by <f(Q)>^2 at each Q, over [Qmin, Qmax] with the Q window
    (:func:`real_space_from_sq`), the form a measured G(r) is reduced to;
    None when no Qmax was given or the Q grid does not cover [Qmin, Qmax],
    with the reason in ``notes``. With r_window 'Lorch' that S(Q) carries
    M(r), so ``pdf_g_from_sq`` is M(r) G(r) terminated, not G(r)
    terminated: G(r) damped towards r_max.
    """

    radiation: str
    weight_unit: str
    s: Series
    f_reduced: Series
    f_keen: Series
    pdf_g: Series
    keen_g: Series
    keen_d: Series
    keen_t: Series
    weights_q0: Mapping[tuple[str, str], float]
    b_mean_sq_q0: float
    fsdp: Mapping[str, Scalar] | None = None
    pdf_g_from_sq: Series | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class ScatteringResult:
    """Everything :func:`analyse_trajectory` measured, with its provenance."""

    provenance: Provenance
    elements: tuple[str, ...]
    concentrations: Mapping[str, float]
    rho0: Scalar
    partial_g: Mapping[tuple[str, str], Series]
    partial_s: Mapping[tuple[str, str], Series]
    totals: Mapping[str, TotalScattering]
    bhatia_thornton: Mapping[str, Series] | None
    notes: tuple[str, ...] = ()


def _tidy(value: float) -> float:
    """A grid end for the export header, without the last-digit noise of
    np.arange (17.000000000000004 -> 17.0)."""
    return float(f"{float(value):.12g}")


def _frame_list(frames: Iterable[int]) -> str:
    return ", ".join(str(k) for k in frames)


def _frame_indices(trajectory, frames) -> list[int]:
    """The chosen readable-frame indices, checked; none is coerced."""
    if frames is None:
        return list(range(trajectory.n_frames))
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames needs a sequence of frame indices")
    indices = []
    for k in frames:
        if isinstance(k, (bool, np.bool_)) or \
                not isinstance(k, (int, np.integer)):
            raise ValueError(f"frame index {k!r} is not a whole number")
        indices.append(int(k))
    return indices


def analyse_trajectory(trajectory, *, r_max_ang: float, dr_ang: float,
                       q_inv_ang, r_window: str, radiations: Sequence[str],
                       frames: Sequence[int] | None = None,
                       q_max_inv_ang: float | None = None,
                       q_min_inv_ang: float = 0.0,
                       q_window: str | None = None,
                       fsdp_window_inv_ang: tuple[float, float] | None = None,
                       fsdp_baseline: str | None = None,
                       neutron_lengths_fm: Mapping[str, float] | None = None,
                       lengths_source: str | None = None,
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None
                       ) -> ScatteringResult:
    """Partial and total scattering of every chosen frame, averaged over frames.

    Per frame: one pair search (:func:`frame_partials`), the partial S_ab(Q)
    by the sine route, and per radiation the total S(Q), F(Q), F_K(Q) and the
    real-space G(r), G_K(r), D(r), T(r) (terminated when ``q_max_inv_ang`` is
    given, with ``q_window`` then required, and then also the G(r)
    transformed from S(Q), ``pdf_g_from_sq``); the Bhatia-Thornton functions
    for a binary model; the FSDP of each total when ``fsdp_window_inv_ang``
    is given (with ``fsdp_baseline`` then required). ``frames``: readable
    frame indices (whole numbers), all of them when None.
    ``neutron_lengths_fm`` / ``lengths_source``: as in
    :func:`scattering_lengths`. A Q grid or a Qmax beyond pi / dr is refused
    up front, as are the FSDP arguments and an element without a factor, so
    no pair search is spent on a run that would stop. A frame that cannot
    be read, or whose box is too small for r_max, is skipped and its reason
    recorded. ``progress(done, total)`` and ``cancelled()`` are plain
    callables for a caller's worker thread; a cancel stops after the current
    frame and the frames not analysed are recorded.
    """
    q = _axis(q_inv_ang, "q_inv_ang")
    r, dr = _r_grid(r_max_ang, dr_ang)
    r_window = _choice(r_window, "r_window", R_WINDOWS)
    radiations = _radiation_list(radiations)
    if not radiations:
        raise ValueError(f"radiations needs at least one of {RADIATIONS}")
    q_max, q_min, q_win = _termination_args(q_max_inv_ang, q_min_inv_ang,
                                            q_window)
    _nyquist_q(float(q[-1]), dr, "q_inv_ang")
    if q_max is not None:
        _nyquist_q(q_max, dr, "q_max_inv_ang")
    if fsdp_window_inv_ang is not None:
        fsdp_window = _range_pair(fsdp_window_inv_ang, "fsdp_window_inv_ang")
        if fsdp_baseline is None:
            raise ValueError(f"fsdp_baseline is needed with fsdp_window_inv_ang: "
                             f"one of {FSDP_BASELINES}")
        fsdp_baseline = _choice(fsdp_baseline, "fsdp_baseline", FSDP_BASELINES)
    else:
        fsdp_window = None
        if fsdp_baseline is not None:
            raise ValueError("fsdp_baseline was given without "
                             "fsdp_window_inv_ang")
    _neutron_overrides(neutron_lengths_fm, lengths_source,
                       list(neutron_lengths_fm or ()))
    overrides = dict(neutron_lengths_fm=neutron_lengths_fm,
                     lengths_source=lengths_source)
    indices = _frame_indices(trajectory, frames)
    if not indices:
        raise ValueError("no frames were chosen")
    outside = [k for k in indices if not 0 <= k < trajectory.n_frames]
    if outside:
        raise ValueError(f"frame(s) {outside} are outside the "
                         f"{trajectory.n_frames} readable frames")
    if len(set(indices)) != len(indices):
        raise ValueError("a frame index is chosen more than once")
    # the factor tables, checked on the first chosen frame's elements before
    # any pair search; a frame that cannot be read is left to the loop
    try:
        first_species = tuple(trajectory.frame(indices[0]).species)
    except (FrameError, ValueError):
        first_species = None
    if first_species:
        for radiation in radiations:
            scattering_lengths(first_species, radiation, **overrides)

    kernel = _sine_kernel(q, r, dr, r_window)
    lorch = _lorch_matrix(r, q_min, q_max) if q_win == "Lorch" else None
    to_g = None
    if q_max is None:
        to_g_note = ("pdf_g_from_sq is not computed: it needs q_max_inv_ang "
                     "and q_window")
    else:
        try:
            to_g = _sq_to_g_matrix(q, r, q_min, q_max, q_win)
            to_g_note = None
        except ValueError as error:
            to_g_note = f"pdf_g_from_sq is not computed: {error}"
    lengths_q: dict[str, dict] = {}
    lengths_0: dict[str, dict] = {}
    factor_notes: dict[str, list[str]] = {}
    real_notes: dict[str, tuple[str, ...]] = {}
    route_notes: tuple[str, ...] = ()
    widths: dict[str, float] = {}
    single: list[str] = []
    given: dict[str, float] = {}

    used: list[int] = []
    skipped: dict[int, str] = {}
    rows: dict[str, list[np.ndarray]] = {}
    rho0_rows: list[float] = []
    fsdp_rows: dict[str, dict[str, list[float]]] = {
        name: {"position": [], "height": [], "fwhm": []} for name in radiations}
    fsdp_why: dict[str, dict[str, list[int]]] = {
        name: {} for name in radiations}
    below_frames: dict[str, list[int]] = {name: [] for name in radiations}
    elements: tuple[str, ...] | None = None
    conc: dict[str, float] = {}
    n_below = 0
    cancelled_at = None

    def keep(key: str, values) -> None:
        rows.setdefault(key, []).append(np.asarray(values, dtype=np.float64))

    for done, k in enumerate(indices):
        if cancelled is not None and cancelled():
            cancelled_at = done
            break
        try:
            frame = trajectory.frame(k)
            partials = frame_partials(frame, r_max_ang=r_max_ang,
                                      dr_ang=dr_ang)
        except (FrameError, ValueError) as error:
            skipped[k] = str(error)
            if progress is not None:
                progress(done + 1, len(indices))
            continue
        if elements is None:
            elements = partials.elements
            conc = partials.concentrations
            given = _neutron_overrides(neutron_lengths_fm, lengths_source,
                                       elements)
            for radiation in radiations:
                lengths_q[radiation] = scattering_lengths(
                    elements, radiation, q, **overrides)
                lengths_0[radiation] = scattering_lengths(
                    elements, radiation, **overrides)
                factor_notes[radiation] = _factor_notes(
                    elements, radiation,
                    given if radiation == "neutron" else {}, lengths_source)
            route_notes, widths = _route_texts(partials, q, r_window)
            single = [n for n in partials.notes
                      if " has one atom in this frame" in n]
        used.append(k)
        n_below += partials.n_below_d_min
        rho0_rows.append(partials.rho0_per_ang3)
        sq = _partial_sq(partials, kernel)
        for pair in partials.g:
            keep(f"g {_pair_name(pair)}", partials.g[pair])
            keep(f"S {_pair_name(pair)}", sq[pair])
        for radiation in radiations:
            total = _total_from_partials(sq, conc, q, radiation,
                                         lengths_q[radiation])
            if any(n.startswith("S(Q) lies below") for n in total.notes):
                below_frames[radiation].append(k)
            keep(f"{radiation} S", total.s)
            keep(f"{radiation} F", total.f_reduced_inv_ang)
            keep(f"{radiation} F_K", total.f_keen)
            real = _real_space(partials, radiation, lengths_0[radiation],
                               q_max, q_min, q_win, None, lorch,
                               factor_notes[radiation])
            real_notes.setdefault(radiation, real.notes)
            keep(f"{radiation} G", real.pdf_g)
            keep(f"{radiation} G_K", real.keen_g)
            keep(f"{radiation} D", real.keen_d)
            keep(f"{radiation} T", real.keen_t)
            if to_g is not None:
                keep(f"{radiation} G from S",
                     to_g[0] @ (total.s[to_g[1]] - 1.0))
            if fsdp_window is not None:
                peak = fsdp(q, total.s, q_window_inv_ang=fsdp_window,
                            baseline=fsdp_baseline)
                fsdp_rows[radiation]["position"].append(peak.q_peak_inv_ang)
                fsdp_rows[radiation]["height"].append(peak.height)
                fsdp_rows[radiation]["fwhm"].append(peak.fwhm_inv_ang)
                if not all(math.isfinite(v) for v in (
                        peak.q_peak_inv_ang, peak.height, peak.fwhm_inv_ang)):
                    for note in peak.notes:
                        if not note.startswith("baseline through"):
                            fsdp_why[radiation].setdefault(note, []).append(k)
        if len(elements) == 2:
            bt = bhatia_thornton(sq, conc)
            keep("S_NN", bt.s_nn)
            keep("S_NC", bt.s_nc)
            keep("S_CC", bt.s_cc)
        if progress is not None:
            progress(done + 1, len(indices))

    if cancelled_at is not None:
        for k in indices[cancelled_at:]:
            skipped[k] = "not analysed: the run was cancelled"
    if not used:
        reasons = "; ".join(f"frame {k}: {why}" for k, why in skipped.items()
                            if not why.startswith("not analysed: the run"))
        if cancelled_at is not None:
            raise ValueError("the run was cancelled before any frame was "
                             "analysed" + (f" ({reasons})" if reasons else ""))
        raise ValueError("no frame could be analysed"
                         + (f" ({reasons})" if reasons else ""))

    def series(key, axis, axis_name, axis_unit, name, unit, notes=()):
        return Series.from_frames(axis, rows[key], name=name,
                                  axis_name=axis_name, axis_unit=axis_unit,
                                  value_unit=unit, frames=used,
                                  notes=tuple(notes))

    in_r = dict(axis=r, axis_name="r_ang", axis_unit="Å")
    in_q = dict(axis=q, axis_name="q_inv_ang", axis_unit="1/Å")
    pairs = _unordered_pairs(list(elements))
    partial_g = {p: series(f"g {_pair_name(p)}", name=f"g(r) {_pair_name(p)}",
                           unit="1", **in_r) for p in pairs}
    window_note = (f"sine transform of g(r) on [0, {r[-1]:.6g}] Å with "
                   f"r_window '{r_window}'")
    partial_s = {p: series(f"S {_pair_name(p)}",
                           name=f"S(Q) {_pair_name(p)} (Faber-Ziman)",
                           unit="1", notes=(window_note,) + route_notes,
                           **in_q)
                 for p in pairs}
    resolution = (f"measured on the sine route's S(Q), whose F(Q) carries the "
                  f"r-window's convolution of full width at half height "
                  f"{widths['conv_fwhm']:.4g} Å^-1 (r_max {r[-1]:.6g} Å, "
                  f"r_window '{r_window}'); the FSDP width includes it")
    from_sq_note = "" if to_g is None else (
        "transform of each frame's total S(Q), normalised by <f(Q)>^2 at "
        f"each Q, over [{q_min:g}, {q_max:g}] Å^-1 with a {q_win} window "
        "(real_space_from_sq); that S(Q) comes from the sine route, so its "
        "cut at r_max is in it")
    if to_g is not None and r_window == "Lorch":
        from_sq_note += (
            "; with r_window 'Lorch' that route multiplied g(r) - 1 by "
            "M(r) = sin(pi r / r_max) / (pi r / r_max) before the transform, "
            "so this G(r) is M(r) G(r), terminated: the model's G(r) damped "
            "towards r_max, where M reaches 0")
    totals = {}
    for radiation in radiations:
        unit = WEIGHT_UNITS[radiation]
        lengths_note = tuple(factor_notes[radiation]) + tuple(
            _range_notes(radiation, q, lengths_q[radiation]))
        weights, f_mean, _ = faber_ziman_weights(conc, lengths_0[radiation])
        s_notes = [window_note, *route_notes, *lengths_note]
        mean_s = np.mean(np.stack(rows[f"{radiation} S"]), axis=0)
        _, f_mean_q, f2_mean_q = faber_ziman_weights(conc,
                                                     lengths_q[radiation])
        bound = np.broadcast_to(1.0 - f2_mean_q / (f_mean_q * f_mean_q),
                                q.shape)
        s_notes += _below_bound_note(q, mean_s, bound)
        below = below_frames[radiation]
        if below:
            s_notes.append(f"{len(below)} of {len(used)} frame(s) have S(Q) "
                           "below 1 - <f^2>/<f>^2 at some Q (frames "
                           f"{_frame_list(below)})")
        peaks = None
        if fsdp_window is not None:
            label = (f"window {fsdp_window[0]:g}-{fsdp_window[1]:g} 1/Å, "
                     f"baseline '{fsdp_baseline}'")
            why = tuple(f"frame(s) {_frame_list(ks)}: {note}"
                        for note, ks in fsdp_why[radiation].items())
            peak_notes = (label, _FSDP_METHOD, resolution) + why
            values = fsdp_rows[radiation]
            peaks = {
                "position": Scalar(f"FSDP position {radiation}", "1/Å",
                                   values["position"], frames=used,
                                   notes=peak_notes),
                "height": Scalar(f"FSDP height {radiation}", "1",
                                 values["height"], frames=used,
                                 notes=peak_notes),
                "fwhm": Scalar(f"FSDP FWHM {radiation}", "1/Å",
                               values["fwhm"], frames=used, notes=peak_notes),
            }
        real_note = real_notes[radiation]
        from_sq = None
        total_notes = list(lengths_note)
        if to_g is not None:
            from_sq = series(
                f"{radiation} G from S", name=f"G(r) from S(Q) {radiation}",
                unit="1/Å^2", notes=(from_sq_note, window_note), **in_r)
        else:
            total_notes.append(to_g_note)
        tag = radiation
        totals[radiation] = TotalScattering(
            radiation=radiation, weight_unit=unit,
            s=series(f"{tag} S", name=f"S(Q) {tag}", unit="1",
                     notes=s_notes, **in_q),
            f_reduced=series(f"{tag} F", name=f"F(Q) = Q[S(Q)-1] {tag}",
                             unit="1/Å", notes=lengths_note, **in_q),
            f_keen=series(f"{tag} F_K", name=f"F_K(Q) {tag}", unit=unit,
                          notes=lengths_note, **in_q),
            pdf_g=series(f"{tag} G", name=f"G(r) {tag}", unit="1/Å^2",
                         notes=real_note, **in_r),
            keen_g=series(f"{tag} G_K", name=f"G_K(r) {tag}", unit=unit,
                          notes=real_note, **in_r),
            keen_d=series(f"{tag} D", name=f"D(r) {tag}",
                          unit=f"{unit}/Å^2", notes=real_note, **in_r),
            keen_t=series(f"{tag} T", name=f"T(r) {tag}",
                          unit=f"{unit}/Å^2", notes=real_note, **in_r),
            weights_q0={p: float(w) for p, w in weights.items()},
            b_mean_sq_q0=_unit_factor(radiation) * float(f_mean) ** 2,
            fsdp=peaks, pdf_g_from_sq=from_sq, notes=tuple(total_notes))
    bt = None
    if len(elements) == 2:
        a, b = elements
        bt_note = (f"A = {a}, B = {b}; C(q) = c_B rho_A(q) - c_A rho_B(q)",)
        bt = {key: series(key, name=f"{key}(Q)", unit="1", notes=bt_note,
                          **in_q) for key in ("S_NN", "S_NC", "S_CC")}

    notes = [window_note,
             f"r grid {r[0]:.6g} .. {r[-1]:.6g} Å step {dr:g} Å; Q grid "
             f"{q[0]:.6g} .. {q[-1]:.6g} Å^-1, {q.size} points"]
    notes.extend(route_notes)
    notes.extend(single)
    if n_below:
        notes.append(f"{n_below} ordered pairs at d <= 0 Å (coincident atoms) "
                     "were left out of g(r) over all frames")
    if trajectory.box_varies:
        notes.append("the box changes between frames; each frame uses its own "
                     "number density, and the sine-route statements above "
                     "use the first analysed frame's box")
    if skipped:
        notes.append(f"{len(skipped)} chosen frame(s) not analysed; reasons "
                     "in the provenance")
    if len(elements) != 2:
        notes.append("Bhatia-Thornton functions are not computed: they are "
                     "defined for two elements and the model has "
                     f"{len(elements)}")
    if to_g is None:
        notes.append(to_g_note)
    if given and "neutron" not in radiations:
        notes.append("neutron_lengths_fm was given, but no neutron total was "
                     "asked for; the lengths are recorded and not used")
    method = {"r_max_ang": _tidy(r_max_ang), "r_last_ang": _tidy(r[-1]),
              "dr_ang": dr, "r_window": r_window,
              "q_first_inv_ang": _tidy(q[0]), "q_last_inv_ang": _tidy(q[-1]),
              "q_points": int(q.size), "radiations": radiations,
              "q_max_termination_inv_ang": q_max,
              "q_min_termination_inv_ang": q_min if q_max is not None
              else None,
              "q_window": q_win,
              "neutron_lengths_fm": tuple(
                  f"{s} {v:g}" for s, v in sorted(given.items())) or None,
              "lengths_source": lengths_source,
              "scattering_factors": tuple(n for radiation in radiations
                                          for n in factor_notes[radiation])}
    if fsdp_window is not None:
        method["fsdp_window_inv_ang"] = fsdp_window
        method["fsdp_baseline"] = fsdp_baseline
    provenance = Provenance.from_trajectory(
        trajectory, frames_used=used, frames_skipped=skipped,
        method_parameters=method, notes=tuple(notes))
    return ScatteringResult(
        provenance=provenance, elements=elements, concentrations=conc,
        rho0=Scalar("number density", "1/Å^3", rho0_rows, frames=used),
        partial_g=partial_g, partial_s=partial_s, totals=totals,
        bhatia_thornton=bt, notes=tuple(notes))
