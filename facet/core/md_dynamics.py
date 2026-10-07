"""Dynamics of an MD trajectory: diffusion, vibrations, conduction, lifetimes.

A frame tells where the atoms are; a trajectory also tells how they move. What
an MD study of a glass or a melt reports from that motion is a short list:
mean-square displacements and the diffusion coefficients read from them, the
non-Gaussian parameter and the self van Hove function that show heterogeneous
motion, the self intermediate scattering function measured by neutron spin
echo and quasi-elastic scattering, the velocity autocorrelation function and
the vibrational density of states, the ionic conductivity and the Haven ratio,
and how long a bond or a coordination shell survives. This module computes
each of them from FACET's own frame model (:mod:`.md_model`), reports each as
a :mod:`.md_stats` container with its spread, and never fills in a quantity
the file and the user did not give.

WHAT IT NEEDS, AND WHERE EACH INPUT COMES FROM
----------------------------------------------
**A time axis.** Taken, in this order: the file's timesteps times a
``timestep_fs`` the user gives; the file position of each frame times a
``frame_interval_ps`` the user gives; the frame times the file itself carries
(``Trajectory.times_ps``). With none of them the call is refused: no MD
timestep is assumed. When the user's value and the file's own times disagree,
the user's is used and a note gives both spans and their ratio (a LAMMPS
``units real`` file read without its unit style shows a ratio of 1000).
Every lag is a whole number of frames, so the frames have to be evenly spaced
in time; an axis whose frames depart from the even grid by more than
:data:`TIME_SPACING_RTOL` of the spacing is refused, naming the trajectory
frames on either side of the first uneven step (a changed dump interval, a
restart that repeats timesteps, a file block the reader skipped). The file's
own record of the chosen frames is held to whichever axis is used: when the
axis comes from ``frame_interval_ps`` (file positions), the file's timesteps,
where every chosen frame has one, have to increase and to be proportional to
the axis within :data:`TIME_SPACING_RTOL` of a frame interval, so a frame
missing from the file, a restart appended to it or a changed dump interval is
refused rather than counted as one more interval. The file's printed frame
times, rounded as printed, are refused when they put a frame half an interval
or more from where the axis puts it (it then sits at another whole-frame lag)
and noted when they depart by more than :data:`TIME_SPACING_RTOL`. A record
that is the same in every chosen frame (every timestep 0, say) holds no time;
it is noted and not checked, so that ``frame_interval_ps`` still states the
axis for such a file. The
reader's own notes and its unit assumption travel with the tracks into every
result. For a VASP XDATCAR the reader keeps the number after
``configuration=`` as the timestep; whether VASP writes the ionic step there
or a count of written configurations when NBLOCK > 1 is not settled here
(reference to verify), so ``timestep_fs`` on an XDATCAR carries a note and
``frame_interval_ps`` (POTIM x NBLOCK) states the axis without that
assumption.

**Continuous positions.** The file's unwrapped coordinates (LAMMPS ``xu yu
zu``, or ``x y z`` with image flags) when frame 0 carries them, otherwise an
unwrap by continuity: between consecutive frames each atom is taken to have
made the shortest move the periodic box allows. In a fixed box that is the
fractional step less its nearest whole number. In a box that changes (NPT),
the step is the Cartesian difference of the wrapped positions less the
lattice vector of the *new* box nearest to it, u(i+1) = u(i) + dw -
round(dw / L(i+1)) L(i+1): the toroidal-view-preserving (TOR) scheme of
von Bülow, Bullerjahn and Hummer [7], which Bullerjahn et al. [6] recommend
for diffusion coefficients. It adds minimal displacements within the box and
so keeps the dynamics of the wrapped trajectory, where unwrapping by image
counts (their lattice schemes, and the unwrapped coordinates LAMMPS writes)
lets the barostat's rescaling of the box enter the displacements. In a box
that changes, the file's unwrapped coordinates are therefore noted as image-
count unwrapped, and ``unwrap='minimum image'`` gives the TOR scheme. Each
atom is unwrapped as its own particle, which [6] states is the scheme's
domain; the atoms of a molecule are not kept whole.

Continuity cannot see a move of more than half the box: it appears as a
shorter move the other way. Such moves cannot be detected one by one, so
the largest step is measured and the trajectory is refused when any atom's
step exceeds :data:`UNWRAP_STEP_LIMIT` of a box width (half of the
half-width at which aliasing begins; a choice, overridable up to 0.5). The
same limit is applied to the file's own unwrapped coordinates and to
positions handed to :func:`tracks_from_arrays`: there a step of a box width
or more between two frames is what image flags reset in a restart, or
positions wrapped into the box, leave behind (measured 2026-10-06 before
this check existed: an NS3 dump with its image flags reset at frame 10 gave
D_Na 6.4e4 times the intact file's, and wrapped arrays an MSD 2.5 times the
true one, both accepted). In those two routes the limit may be set
above 0.5, as an explicit statement that the coordinates are continuous. The
largest step, in Å and as a fraction of the width, is part of the result.
A file whose atoms are not the same atom from frame to frame (a LAMMPS dump
without an id column, ``ids_track_atoms`` False) is refused: a displacement
between two of its frames has no meaning.

**Velocities**, for the velocity autocorrelation, from the file
(``Frame.vel_ang_per_ps``). Central differences of the unwrapped positions are
used only when the user asks (``velocities='finite difference'``), with a note
that a vibration of frequency nu is then attenuated by sin(2 pi nu dt) /
(2 pi nu dt) in amplitude and the two end frames have no velocity.

**Charges and temperature**, for the conductivity, are required arguments.
Formal oxidation states and force-field partial charges give different
conductivities, and the choice is the user's; the run's temperature is a
property of the run, not of the file. :func:`kinetic_temperature` measures the
kinetic temperature of the frames so the two can be set side by side.

**Masses** are the standard atomic weights in gemmi, used for the centre of
mass, the mass-weighted velocity autocorrelation and the kinetic temperature.
The file's own masses (a LAMMPS data file's ``Masses``) are not read; a force
field may use other masses (isotopes, rounded values), and the notes say
which were used.

TIME AVERAGES, BLOCKS AND THE SPREAD
------------------------------------
Every correlation function is averaged over all time origins inside a block of
consecutive frames and over the atoms of an element. ``n_blocks`` cuts the
frames into that many equal blocks (frames left over at the end are named in
a note, never silently used or lost); each block is one row of the
:mod:`.md_stats` container, and the spread is the sample standard deviation
across blocks. One block, the default, gives every origin pair the trajectory
holds and a NaN spread with a note. A lag is reached only inside one block,
so more blocks give a spread and a shorter reach.

THE QUANTITIES
--------------
* **MSD**, <|r(t0 + t) - r(t0)|^2> over every origin, by the FFT algorithm of
  nMOLDYN [1, 2]: MSD(m) = S1(m) - 2 S2(m), with S2 the position
  autocorrelation from one zero-padded FFT and S1 from running sums of
  r(k)^2. Each atom's track is centred on its own time average first (the
  MSD does not change; the sums it is formed from become smaller, and so do
  their roundings). O(T log T) per atom instead of O(T^2); it equals the
  direct double sum to 1e-10 Å^2 in the tests. MSD(0) is set to 0.
* **Diffusion coefficient**, D = slope / 6 of a straight line fitted by least
  squares to the block-mean MSD over a window the user chooses
  (``t_min_ps``, ``t_max_ps``; no default: where the ballistic and caged
  regimes end depends on the system and the temperature). The result carries
  the window, the number of lags in it, the residuals, the slope of log MSD
  against log t over the window (1 where the motion is diffusive, 2 where it
  is ballistic) and D fitted to each block. No standard error of the slope is
  given: MSD values at neighbouring lags are strongly correlated, and the
  least-squares error formula would treat them as independent [5]. When the
  centre-of-mass drift was not removed, the centre of mass's own MSD is
  fitted over the same window and its slope / 6 is given beside D, with its
  ratio to D: in a 300 K NVT NS3 glass model whose centre of mass moved 0.6 Å
  in 19 ps that ratio was 0.99 for Si over 5-19 ps (measured 2026-10-06),
  while the log-log slope for O was 1.03, a drift the log-log slope alone
  does not show.
* **Centre-of-mass removal** subtracts the centre of mass of the whole model
  (every atom, standard atomic weights) from each position. LAMMPS's
  ``compute msd ... com yes`` subtracts the centre of mass of the compute's
  own group instead, so the two agree only when the group is the whole model
  (the argon check below). For a group of one element they differ: on the NS3
  hot melt (3000 atoms, one origin, 19.8 ps) LAMMPS's ``com yes`` for the Na
  group differed from the whole-model removal by 0.449 Å^2, 2.0e-3 relative,
  and by 1.1e-3 to 1.4e-3 for Si and O, while removing each group's own centre
  of mass reproduced LAMMPS to 6e-6 (measured 2026-10-06 against LAMMPS's own
  output). The whole-model choice keeps the motion of one sublattice against
  the others in that sublattice's MSD.
* **Green-Kubo diffusion coefficient**, D = (1/3) x the integral of
  <v(t0) . v(t0 + t)> from 0 to an upper limit the user gives [18, 19], the
  velocity route to the quantity the MSD route fits; the running integral is
  returned on every lag, so its plateau, or the lack of one, can be read
  beside the MSD's D. For a stationary process the two agree once the VACF
  has decayed: the MSD's slope at long times is 2 x that integral.
* **Non-Gaussian parameter** alpha2(t) = 3 <dr^4> / (5 <dr^2>^2) - 1 [3],
  0 for Gaussian displacements, at lags the user lists; computed per block
  from the pooled moments of the block, then averaged over blocks.
* **Self van Hove function** G_s(r, t) [4], the distribution of |dr| at a
  lag, as a :class:`~.md_stats.Histogram` density 4 pi r^2 G_s (1/Å, on the
  user's edges, samples outside counted) and as G_s itself (1/Å^3, each bin
  divided by its shell volume).
* **Self intermediate scattering function** F_s(q, t) =
  <sin(q |dr|) / (q |dr|)>, the exact average of exp(i q . dr) over the
  directions of q, at the user's q values; exp(-q^2 D t) for Gaussian
  displacements.
* **Distinct van Hove function** G_d(r, t) for element pairs, from the
  positions of A at t0 and of B at t0 + t. The pairs come from
  :func:`.bulk.iter_pairs` run on one frame that holds both sets of
  positions, so no second neighbour search exists in FACET. Each lag uses
  ``n_origins`` evenly spaced origins (a subsampling, stated), each origin a
  row. Normalised by the number of distinct B partners per unit volume,
  (N_B - [A = B]) / V, so that it is the partial g(r) at t = 0 and 1 for
  uncorrelated positions.
* **Velocity autocorrelation** <v(t0) . v(t0 + t)> per element, by FFT, raw
  (Å^2/ps^2) and normalised to 1 at t = 0, and the mass-weighted sum over
  every atom of the tracks.
* **Vibrational density of states** [8], g(nu) = 4 * integral_0^tmax of
  C(t) cos(2 pi nu t) dt with C the normalised VACF, so that the integral of
  g over 0 .. the Nyquist frequency 1/(2 dt) is 1 (exactly, by the
  trapezoid rule on the frequency grid). The integral is the trapezoid rule
  on the lag grid, evaluated as the real FFT of the even extension of C. A
  half-Hann taper (``window='hann'``, a smoothing choice, stated) takes C to 0
  at the last lag; ``window='none'`` leaves it as measured. Frequencies in THz
  (1/ps) and in cm^-1, converted with ``scipy.constants`` (c, tera, centi).
  A vibration above the Nyquist frequency is not removed by sampling every
  dt: it folds back (aliases) into 0 .. 1/(2 dt), and the normalisation keeps
  its power there. The normalised VACF at the first lag, C(dt), is measured
  and reported for that reason: on an NS3 hot melt dumped every 0.2 ps it was
  -0.02 to -0.06 (0.86 to 0.99 from the same run dumped every 5 fs), and the
  whole VDOS area then sat below the 2.5 THz Nyquist frequency, where the
  5 fs dump puts 4 to 26 % of it (measured 2026-10-06).
* **Ionic conductivity**, two ways. Nernst-Einstein from the tracer
  diffusion coefficients, sigma = e^2 / (V k_B T) sum_a N_a z_a^2 D_a, which
  leaves out every cross-correlation between ions. Collective, from the
  charge displacement M(t) = sum_i q_i r_i(t) [9]: sigma = e^2 / (6 V k_B T)
  times the slope of <|M(t0 + t) - M(t0)|^2>, fitted over the user's window.
  Their ratio sigma_NE / sigma is the Haven ratio [10], the ratio of the two
  block-mean conductivities; each block's own ratio is given beside it when
  both were measured over the same blocks of frames (every result carries
  its blocks' first and last frames, and blocks are never paired by index
  alone), and the mean of those is not H_R. Units S/m; e and k_B from
  ``scipy.constants``. The diffusion coefficients enter Nernst-Einstein as
  fit objects, which carry their unit, or as bare numbers only under
  ``d_m2_per_s``: the module reports D in Å^2/ps as well, and a value in
  Å^2/ps read as m^2/s gives a conductivity 1e8 times too large. Elements
  charged in the collective sum and absent from the Nernst-Einstein one, and
  charges given for elements the tracks do not hold, are named in the notes.
* **Bond lifetimes.** A bond is a cation-anion pair bonded at the frame's
  bond-valence threshold (:func:`.bulk.bonds_at`, the bond definition of the
  CN and of every glass descriptor). With h(t) = 1 while the pair is bonded,
  the intermittent function C_I(t) = <h(t0) h(t0 + t)> / <h(t0)> and the
  continuous function C_C(t) = <h(t0) H(t0, t0 + t)> / <h(t0)>, H = 1 when the
  bond held at every frame in between [11, 12, 13], both pooled over the
  bonds of each element pair and every origin. Breaks shorter than the frame
  interval cannot be seen, so C_C depends on the dump interval; the notes say
  so. ``gap_tolerance_frames`` bridges breaks of up to that many frames
  before the continuous function is formed, after the transient-excursion
  idea of Impey, Madden and McDonald [14]. The convention here is the filled
  history: each interior break of up to that many frames is set to bonded,
  and C_C and its origin count are then formed on the filled history, so a
  bridged frame counts as bonded wherever it falls, at an origin and at an
  end point alike. Another reading, a bond present at t0 and at t0 + t with
  no excursion longer than the tolerance between, counts only origins and end
  points where the bond is present, and gives other values: on h = 1 0 1 1 0
  0 1 with a tolerance of 1 frame, 1, 0.75, 0.5, 0.25 here against 1, 1/3,
  1/3, 1/3 for that reading. Which one [14] uses is not settled here
  (reference to verify: the 1983 text was not obtained). The mean residence
  time is the integral of C(t) or the time constant of an exponential fitted
  to it over a window; which one, and the window, are the user's and are
  stated. C_I does not decay to 0 while bonds that break form again: for
  presence uncorrelated in time it sits at sum_i p_i^2 / sum_i p_i, p_i the
  fraction of a block's frames bond i is present. That level is measured per
  block and reported, and the integral of C_I, which then grows with the
  upper limit, carries a note saying so with both values.

WHAT IS NOT HERE
----------------
No physical parameter has a default: not the timestep, not a charge, not the
temperature, not a fit window. Method parameters (``n_blocks``, the lag list,
the histogram edges, the VDOS taper, ``gap_tolerance_frames``,
:data:`UNWRAP_STEP_LIMIT`) are arguments, stated in each result's notes and
``method_parameters``. The tests are synthetic trajectories with known
answers (``tests/test_md_dynamics.py``); the outside checks are the LAMMPS
liquids below, and on 2026-10-06 and 07 the module was also run on LAMMPS
dumps of a 3000-atom NS3 glass at 300 K and of its melt (against LAMMPS's
own ``compute msd``, ``vacf`` and ``rdf`` and its thermo temperature, and on
copies damaged on purpose: a frame removed, a restart appended, image flags
reset), which is where the centre-of-mass, aliasing, time-record and
image-flag checks described above come from.

CHECKED AGAINST OUTSIDE IMPLEMENTATIONS
---------------------------------------
Two Lennard-Jones argon liquids (864 atoms, LAMMPS units metal, epsilon
0.0104 eV, sigma 3.4 Å, cut at 8.5 Å, 120 K; 501 frames 0.05 ps apart), one
NVE and one NPT at 200 bar, were run with LAMMPS (22 Jul 2025) [16] in a
throwaway environment outside FACET, dumped with ``x y z``, ``xu yu zu`` and
``vx vy vz`` at 15 significant figures, and read through
``md_readers.read_trajectory``. LAMMPS and tidynamics [17] are not FACET
dependencies, and no test imports them. Measured 2026-10-06:

* LAMMPS's own ``compute msd`` (one origin; ``com no`` and ``com yes``) and
  ``compute vacf`` against the same single-origin quantities formed from the
  tracks: 1.1e-14 relative and 8.9e-15 Å^2/ps^2 at most.
* tidynamics 1.1.2 (``msd`` and ``acf``, FFT over every origin), on the
  dump's columns parsed by separate code: the MSD within 1.5e-10 Å^2 (of up to
  85 Å^2), the VACF within 9.4e-14 Å^2/ps^2. D from ``numpy.polyfit`` on
  tidynamics' MSD over 2-10 ps equals :func:`fit_diffusion`'s to six figures
  (0.276684 Å^2/ps, 2.77e-9 m^2/s, NVE).
* alpha2 and F_s at four lags against plain numpy on the dump's ``xu``:
  6.7e-16 and 1.1e-16 (NVE).
* The unwrap by continuity against LAMMPS's ``xu`` in NVE: 1.6e-13 Å. In NPT
  the two differ by up to 3.1 Å after 25 ps, the TOR-versus-image-count
  difference of [6]: the MSDs by up to 0.63 Å^2, D 0.576073 (TOR) against
  0.577181 Å^2/ps (``xu``).
* :func:`kinetic_temperature` times 3N / (3N - 3) against LAMMPS's ``temp``
  (which removes 3 degrees of freedom): -1.1944010e-6 relative in every
  frame, which is exactly the product of LAMMPS's metal-unit constants
  (``boltz`` 8.617343e-5 eV/K, 1.13e-6 above CODATA 2022's k_B; ``mvv2e``
  1.0364269e-4, 6.4e-8 below the value from CODATA's amu and e) over
  scipy's: predicted -1.1944010e-6.

TIMINGS
-------
Measured 2026-10-06 on Windows 11, i5-13420H, Python 3.11.9, numpy 2.4.6,
scipy 1.15.1, unpinned, while another session's LAMMPS run held about seven
of the twelve logical CPUs: :func:`.bulk.analyse_frame` on the same 10 648-
atom cubic frame, timed at the start and the end as a reference, took 2.1
and 2.5 s (median of 5) against the 1.10 s ``bulk.py`` records unpinned for
that box, so the figures below ran about twice as long as on a quiet
machine. The model is
``tools/bench_md.make_box(22)`` (10 648 atoms, Si/O/Na on a jittered grid)
moved as a random walk; medians of 3 runs, single runs where marked (1)::

                                             200 frames     1 000 frames
    tracks_from_arrays (1)                     1.7 s           6.1 s
    msd, 3 elements + centre of mass           1.7 s           6.9 s
    fit_diffusion                              0.01 s          0.002 s
    self_correlations, 10 lags (1)             3.5 s           8.2 s
      + 100-bin van Hove and 3 q values (1)   11.4 s          44.4 s
    vacf, 3 elements + mass-weighted total     1.3 s           7.2 s
    vdos                                       0.02 s          0.02 s
    kinetic_temperature                        0.14 s          1.1 s
    collective_conductivity                    0.07 s          0.41 s
    bond_lifetimes, 20 000 bonds (1)           1.2 s           5.4 s
      with gap_tolerance_frames = 2 (1)        1.0 s           7.5 s

    distinct_van_hove, one lag, one origin, r <= 8 Å: Na-Na (1 568 Na)
    0.34 s; O-O (6 379 O, 12 758 atoms in the two-time frame) 7.4 s (1)
    collect_tracks by continuity, 50 frames held as Frame objects: 0.20 s (1)
    collect_bonds, 5 frames: 12.4 s (1), 2.5 s a frame, the reference's
    bulk.analyse_frame and nothing more

Measured again 2026-10-07 after the step check was added to
:func:`tracks_from_arrays` (and :func:`green_kubo_diffusion`, the centre-of-
mass fit in :func:`fit_diffusion` and the file-record check of the time axis
were added), same model and machine under heavier load: the reference took
7.3 and 8.7 s (median of 5, start and end), about 3.5 times the run above.
Medians of 3::

                                             200 frames     1 000 frames
    tracks_from_arrays (with the step check)   3.4 s          15.5 s
    fit_diffusion, with the centre-of-mass fit 0.011 s        0.024 s
    green_kubo_diffusion, 3 elements           0.016 s        0.015 s
    time_axis_ps with timesteps and times      -              0.03 s

The step check forms the fractional steps as d @ inv(box) over chunks of
frames; one ``numpy.linalg.solve`` per frame, the earlier layout, took 9.1 to
10.1 s against 4.4 to 6.6 s for the chunks on 10 648 atoms x 1 000 frames
(interleaved, one process, the same load).

Memory: the tracks hold 24 bytes per atom per frame for the positions and as
much for the velocities (256 MB each at 10 648 atoms x 1 000 frames); every
FFT works on chunks of :data:`CHUNK_VALUES`. The self-correlation pass with
the van Hove histogram and three q values cost 0.51 µs per displacement at
1 000 frames (lags 1 to 999 frames, log-spaced: 8.7e7 displacements) and
0.69 µs at 200 frames (1.7e7), under the load above. An earlier layout that
ran the FFT along the first (time) axis was 25-30 % slower on 10 648 atoms x
1 000 frames, measured interleaved in one process; the mass-weighted VACF is
formed from the element sums, so the VACF takes one FFT pass, not two.

REFERENCES
----------
[1] G. R. Kneller, V. Keiner, M. Kneller and M. Schiller, "nMOLDYN: A program
    package for a neutron scattering oriented analysis of Molecular Dynamics
    simulations", *Computer Physics Communications* 91 (1995) 191-214,
    https://doi.org/10.1016/0010-4655(95)00048-K
[2] V. Calandrini, E. Pellegrini, P. Calligari, K. Hinsen and G. R. Kneller,
    "nMoldyn - Interfacing spectroscopic experiments, molecular dynamics
    simulations and models for time correlation functions", *École
    thématique de la Société Française de la Neutronique* 12 (2011) 201-232,
    https://doi.org/10.1051/sfn/201112010
[3] A. Rahman, "Correlations in the Motion of Atoms in Liquid Argon",
    *Physical Review* 136 (1964) A405-A411,
    https://doi.org/10.1103/PhysRev.136.A405
[4] L. Van Hove, "Correlations in Space and Time and Born Approximation
    Scattering in Systems of Interacting Particles", *Physical Review* 95
    (1954) 249-262, https://doi.org/10.1103/PhysRev.95.249
[5] M. P. Allen and D. J. Tildesley, *Computer Simulation of Liquids*, 2nd
    ed., Oxford University Press, Oxford (2017),
    https://doi.org/10.1093/oso/9780198803195.001.0001
[6] J. T. Bullerjahn, S. von Bülow, M. Heidari, J. Hénin and G. Hummer,
    "Unwrapping NPT Simulations to Calculate Diffusion Coefficients",
    *Journal of Chemical Theory and Computation* 19 (2023) 3406-3417,
    https://doi.org/10.1021/acs.jctc.3c00308
[7] S. von Bülow, J. T. Bullerjahn and G. Hummer, "Systematic errors in
    diffusion coefficients from long-time molecular dynamics simulations at
    constant pressure", *The Journal of Chemical Physics* 153 (2020) 021101,
    https://doi.org/10.1063/5.0008316
[8] J. M. Dickey and A. Paskin, "Computer Simulation of the Lattice Dynamics
    of Solids", *Physical Review* 188 (1969) 1407-1418,
    https://doi.org/10.1103/PhysRev.188.1407
[9] E. Helfand, "Transport Coefficients from Dissipation in a Canonical
    Ensemble", *Physical Review* 119 (1960) 1-9,
    https://doi.org/10.1103/PhysRev.119.1
[10] G. Murch, "The Haven ratio in fast ionic conductors", *Solid State
    Ionics* 7 (1982) 177-198, https://doi.org/10.1016/0167-2738(82)90050-9
[11] D. C. Rapaport, "Hydrogen bonds in water", *Molecular Physics* 50 (1983)
    1151-1162, https://doi.org/10.1080/00268978300102931
[12] A. Luzar and D. Chandler, "Hydrogen-bond kinetics in liquid water",
    *Nature* 379 (1996) 55-57, https://doi.org/10.1038/379055a0
[13] A. Luzar, "Resolving the hydrogen bond dynamics conundrum", *The Journal
    of Chemical Physics* 113 (2000) 10663-10675,
    https://doi.org/10.1063/1.1320826
[14] R. W. Impey, P. A. Madden and I. R. McDonald, "Hydration and mobility of
    ions in solution", *The Journal of Physical Chemistry* 87 (1983)
    5071-5083, https://doi.org/10.1021/j150643a008
[15] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific
    computing in Python", *Nature Methods* 17 (2020) 261-272,
    https://doi.org/10.1038/s41592-019-0686-2 (``scipy.constants`` holds the
    CODATA values; CODATA 2022 in the scipy 1.15.1 installed here, where e and
    k_B are exact by the 2019 SI definitions).
[16] A. P. Thompson et al., "LAMMPS - a flexible simulation tool for
    particle-based materials modeling at the atomic, meso, and continuum
    scales", *Computer Physics Communications* 271 (2022) 108171,
    https://doi.org/10.1016/j.cpc.2021.108171
[17] P. de Buyl, "tidynamics: A tiny package to compute the dynamics of
    stochastic and molecular simulations", *Journal of Open Source Software*
    3 (2018) 877, https://doi.org/10.21105/joss.00877
[18] M. S. Green, "Markoff Random Processes and the Statistical Mechanics of
    Time-Dependent Phenomena. II. Irreversible Processes in Fluids", *The
    Journal of Chemical Physics* 22 (1954) 398-413,
    https://doi.org/10.1063/1.1740082
[19] R. Kubo, "Statistical-Mechanical Theory of Irreversible Processes. I.
    General Theory and Simple Applications to Magnetic and Conduction
    Problems", *Journal of the Physical Society of Japan* 12 (1957) 570-586,
    https://doi.org/10.1143/JPSJ.12.570
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import bv
from .md_model import (NO_TIMESTEP, FrameError, ModelOxidation, _checked_box,
                       frame_from_arrays, validate_symbol)
from .md_stats import Histogram, Scalar, Series

__all__ = [
    "TIME_SPACING_RTOL", "UNWRAP_STEP_LIMIT", "CHUNK_VALUES", "UNWRAP_METHODS",
    "VELOCITY_SOURCES", "WINDOWS", "RESIDENCE_METHODS", "BOND_KINDS",
    "time_axis_ps", "AtomTracks", "collect_tracks", "tracks_from_arrays",
    "log_spaced_lag_times",
    "MSDResult", "msd", "DiffusionFit", "fit_diffusion",
    "SelfCorrelations", "self_correlations",
    "DistinctVanHove", "distinct_van_hove",
    "VACFResult", "vacf", "VDOSResult", "vdos", "GreenKuboDiffusion",
    "green_kubo_diffusion", "kinetic_temperature",
    "NernstEinstein", "nernst_einstein", "CollectiveConductivity",
    "collective_conductivity", "HavenRatio", "haven_ratio",
    "BondTimeline", "collect_bonds", "bond_timeline", "BondLifetimes",
    "bond_lifetimes", "ResidenceTime", "residence_time",
]

# ---------------------------------------------------------------------------
# constants: each one is a choice, and says so
# ---------------------------------------------------------------------------

# How far a frame time may sit from the even grid, as a fraction of the frame
# spacing. A numerical tolerance: times read back from text sit off the grid
# by their printing rounding, accepted while it stays under a thousandth of
# the spacing; a dump interval changed by one timestep in a hundred moves the
# frames after it by a hundredth of the spacing or more, and is refused.
TIME_SPACING_RTOL: float = 1e-3

# The largest step between consecutive frames, as a fraction of the box width
# perpendicular to a face, that the unwrap by continuity accepts. A detection
# choice, not a physical limit: a true step beyond half the width appears as
# a shorter step the other way and cannot be seen, so steps are refused well
# before that point (here at half of the half-width).
UNWRAP_STEP_LIMIT: float = 0.25

# About how many float64 values one working array holds (32 MB). A memory
# choice: the atoms are processed in chunks of this size, and no result
# depends on it beyond the rounding of sums.
CHUNK_VALUES: int = 4_000_000

UNWRAP_METHODS = ("auto", "file", "minimum image")
VELOCITY_SOURCES = ("file", "finite difference")
WINDOWS = ("hann", "none")
RESIDENCE_METHODS = ("integral", "exponential fit")
BOND_KINDS = ("continuous", "intermittent")


# ---------------------------------------------------------------------------
# small checks
# ---------------------------------------------------------------------------

def _constants():
    import scipy.constants as constants

    return constants


def _fft():
    import scipy.fft as sfft

    return sfft


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} {value!r} is not a number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} {value!r} is not a number") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} is {value!r}; a finite number is needed")
    return out


def _positive(value, name: str, *, allow_zero: bool = False) -> float:
    out = _number(value, name)
    if out < 0.0 or (out == 0.0 and not allow_zero):
        raise ValueError(f"{name} is {value!r}; a number "
                         f"{'of 0 or more' if allow_zero else 'above 0'} is "
                         "needed")
    return out


def _whole(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or \
            not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} {value!r} is not a whole number")
    if int(value) < minimum:
        raise ValueError(f"{name} is {value}; {minimum} or more is needed")
    return int(value)


def _choice(value, name: str, options: tuple[str, ...]) -> str:
    if value not in options:
        raise ValueError(f"{name} {value!r} is not one of {options}")
    return value


def _frozen(array) -> np.ndarray:
    out = np.array(array, copy=True)
    out.setflags(write=False)
    return out


def _unique_notes(notes) -> tuple[str, ...]:
    out: list[str] = []
    for note in notes:
        if note and note not in out:
            out.append(note)
    return tuple(out)


def _float_list(value) -> str:
    return ", ".join(f"{v:.6g}" for v in value)


def _weights_amu(symbols: np.ndarray) -> np.ndarray:
    """Standard atomic weights from gemmi, one per atom."""
    import gemmi

    tokens, inverse = np.unique(np.asarray(symbols), return_inverse=True)
    weights = np.array([float(gemmi.Element(str(t)).weight) for t in tokens])
    if (weights <= 0).any():
        raise ValueError(
            "gemmi has no atomic weight for "
            + ", ".join(str(t) for t, w in zip(tokens, weights) if w <= 0))
    return weights[inverse.reshape(-1)]


def _masses_note(symbols) -> str:
    tokens = np.unique(np.asarray(symbols))
    weights = _weights_amu(tokens)
    return ("masses are gemmi's standard atomic weights ("
            + ", ".join(f"{t} {w:g} amu" for t, w in zip(tokens, weights))
            + "); the file's own masses are not read")


def _even_spacing(t_ps: np.ndarray, what: str,
                  labels: Sequence[int] | None = None) -> float:
    """The spacing of an evenly spaced time axis, or ValueError saying where
    it is not.

    ``labels`` are the trajectory frame indices of the times; the refusal
    names the two frames on either side of the first step that differs from
    the median step by more than :data:`TIME_SPACING_RTOL` of it (or, for a
    slow drift with no such step, the first frame off the even grid)."""
    t = np.asarray(t_ps, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"{what}: one time per frame is needed")
    if t.size < 2:
        raise ValueError(f"{what}: a time series needs at least 2 frames, not "
                         f"{t.size}")
    if not np.isfinite(t).all():
        raise ValueError(f"{what}: a frame time is NaN or infinite")
    if labels is None:
        def name(k: int) -> str:
            return f"position {k}"
    else:
        tags = [int(v) for v in labels]
        if len(tags) != t.size:
            raise ValueError(f"{what}: one frame label per time is needed")

        def name(k: int) -> str:
            return f"frame {tags[k]}"
    steps = np.diff(t)
    if (steps <= 0).any():
        k = int(np.flatnonzero(steps <= 0)[0])
        raise ValueError(
            f"{what}: the times do not increase: {name(k + 1)} is at "
            f"{float(t[k + 1])!r} ps after {name(k)} at {float(t[k])!r} ps (a "
            "restart that repeats timesteps, or frames out of order); a range "
            "in which time increases is needed")
    dt = (t[-1] - t[0]) / (t.size - 1)
    deviation = np.abs(t - (t[0] + dt * np.arange(t.size))) / dt
    if deviation.max() > TIME_SPACING_RTOL:
        typical = float(np.median(steps))
        odd = np.flatnonzero(np.abs(steps - typical)
                             > TIME_SPACING_RTOL * typical)
        if odd.size:
            k = int(odd[0])
            where = (f"from {name(k)} at {float(t[k])!r} ps to {name(k + 1)} "
                     f"at {float(t[k + 1])!r} ps the step is "
                     f"{float(steps[k]):.6g} ps, where the median step is "
                     f"{typical:.6g} ps (they differ by more than "
                     f"TIME_SPACING_RTOL = {TIME_SPACING_RTOL:g} of it)")
        else:
            k = int(np.flatnonzero(deviation > TIME_SPACING_RTOL)[0])
            where = (f"{name(k)} at {float(t[k])!r} ps lies "
                     f"{deviation[k] * dt:.6g} ps from the even grid of "
                     f"{dt:.6g} ps (more than TIME_SPACING_RTOL = "
                     f"{TIME_SPACING_RTOL:g} of the spacing)")
        spacings = np.unique(np.round(steps, 9))
        raise ValueError(
            f"{what}: the frames are not evenly spaced in time: {where}; "
            f"spacings found: {_float_list(spacings[:6])} ps"
            + (" ..." if spacings.size > 6 else "")
            + ". Lags are counted in whole frames, so the frames need one "
            "interval throughout (a changed dump interval, or a file block the "
            "reader skipped, breaks it); a range with one interval can be "
            "chosen, and where the file prints its times coarsely, "
            "timestep_fs or frame_interval_ps gives an axis without that "
            "rounding")
    return float(dt)


def _file_record_check(t_ps: np.ndarray, record: np.ndarray,
                       labels: Sequence[int], what: str, *, exact: bool,
                       axis: str) -> list[str]:
    """Hold the file's own record of the chosen frames (timesteps or printed
    times) to the time axis ``t_ps`` built from another source.

    The record has to increase from frame to frame, and to be proportional to
    the axis: with s the median ratio of the axis step to the record step,
    each frame's deviation from t0 + s (record - record0) is measured in
    frame intervals. An ``exact`` record (integer timesteps against an axis
    with no printing rounding) is refused beyond :data:`TIME_SPACING_RTOL`;
    a printed one is refused at half an interval or more (the frame then
    sits at another whole-frame lag) and noted beyond
    :data:`TIME_SPACING_RTOL`. Returns the notes."""
    axis_ps = np.asarray(t_ps, dtype=np.float64)
    rec = np.asarray(record, dtype=np.float64)    # timesteps, or ps
    if axis_ps.size < 2 or not np.isfinite(rec).all():
        return []
    tags = [int(v) for v in labels]
    unit = "" if what == "timesteps" else " ps"
    one = "timestep" if what == "timesteps" else "time"
    if (rec == rec[0]).all():
        # a record that never moves holds no time to check; refusing it as
        # "not increasing" left frame_interval_ps, the one way to state the
        # axis for such a file, refused as well
        return [f"every chosen frame carries the same {one} in the file "
                f"({rec[0]:.10g}{unit}), so the file's {what} hold no time "
                f"and are not held to the time axis from {axis}"]
    steps = np.diff(rec)
    axis_steps = np.diff(axis_ps)
    if (steps <= 0).any():
        k = int(np.flatnonzero(steps <= 0)[0])
        # the axis's step there, measured (an earlier wording said "one
        # interval later" whatever the axis was)
        raise ValueError(
            f"the file's own {what} do not increase: frame {tags[k + 1]} has "
            f"{one} {rec[k + 1]:.10g}{unit} after frame {tags[k]} at "
            f"{rec[k]:.10g}{unit} (a restart appended to the file, or frames "
            f"out of order), while the time axis from {axis} places it "
            f"{float(axis_steps[k]):.6g} ps later; a range of frames over "
            f"which the file's {what} increase can be chosen")
    if (axis_steps <= 0).any():
        return []                       # _even_spacing refuses it, by name
    spacing = (axis_ps[-1] - axis_ps[0]) / (axis_ps.size - 1)
    scale = float(np.median(axis_steps / steps))
    deviation = np.abs(axis_ps - (axis_ps[0] + scale * (rec - rec[0]))) \
        / spacing
    limit = TIME_SPACING_RTOL if exact else 0.5
    if deviation.max() > limit:
        k = int(np.flatnonzero(deviation > limit)[0])
        typical = float(np.median(steps))
        # the axis's own step there, measured: one interval when the axis is
        # file positions x frame_interval_ps, anything when it is the file's
        # printed times (an earlier wording said "one interval" in both)
        raise ValueError(
            f"the file's own {what} disagree with the time axis from {axis}: "
            f"from frame {tags[k - 1]} ({one} {rec[k - 1]:.10g}{unit}) to "
            f"frame {tags[k]} ({one} {rec[k]:.10g}{unit}) the file records a "
            f"step of {rec[k] - rec[k - 1]:.10g}{unit} where the median step "
            f"of the chosen frames is {typical:.10g}{unit}, and the axis "
            "steps by "
            f"{float(axis_steps[k - 1]):.6g} ps where its median step is "
            f"{float(np.median(axis_steps)):.6g} ps; frame {tags[k]} lies "
            f"{deviation[k]:.4g} of a frame interval from where the file's own "
            "record puts it (more "
            + ("than TIME_SPACING_RTOL = " f"{TIME_SPACING_RTOL:g}" if exact
               else "than half an interval")
            + "): a frame missing from the file, frames written twice, or a "
            "changed dump interval. A range of frames with one interval can "
            "be chosen"
            + ("" if what != "timesteps" else
               ", and timestep_fs builds the axis from the timesteps"))
    if deviation.max() > TIME_SPACING_RTOL:
        k = int(np.flatnonzero(deviation > TIME_SPACING_RTOL)[0])
        return [f"the file's own {what} depart from the time axis from "
                f"{axis} by up to {deviation.max():.4g} of a frame interval "
                f"(first beyond TIME_SPACING_RTOL = {TIME_SPACING_RTOL:g} at "
                f"frame {tags[k]}): printing rounding, or a dump interval "
                "that changed by less than half an interval; the axis from "
                f"{axis} is used"]
    return []


def _frame_list(trajectory, frames) -> list[int]:
    if frames is None:
        return list(trajectory.frame_indices())
    if isinstance(frames, (str, bytes)):
        raise ValueError("frames needs a sequence of frame indices")
    out = []
    for k in frames:
        if isinstance(k, (bool, np.bool_)) or \
                not isinstance(k, (int, np.integer)):
            raise ValueError(f"frame index {k!r} is not an integer")
        if not 0 <= int(k) < trajectory.n_frames:
            raise ValueError(f"frame {int(k)} is outside 0 .. "
                             f"{trajectory.n_frames - 1}")
        out.append(int(k))
    if len(out) > 1 and not (np.diff(out) > 0).all():
        raise ValueError("frames need to be in increasing order, each once, "
                         "so that time runs forward")
    return out


def _tracking_check(trajectory) -> None:
    if not getattr(trajectory, "ids_track_atoms", True):
        raise ValueError(
            "the atoms of this file are not the same atom from frame to frame "
            "(ids_track_atoms is False: a LAMMPS dump without an id column, "
            "whose rows LAMMPS re-orders between frames), so a displacement "
            "between two frames has no meaning; a dump with 'id' in its "
            "columns is needed for dynamics")


def _load(trajectory, k: int):
    try:
        return trajectory.frame(k)
    except FrameError as error:
        raise ValueError(
            f"frame {k} could not be read ({error}); a time series cannot "
            "leave a frame out without breaking the even lag grid, so the "
            "frames chosen need to avoid it") from error


def time_axis_ps(trajectory, frames: Sequence[int] | None = None, *,
                 timestep_fs: float | None = None,
                 frame_interval_ps: float | None = None
                 ) -> tuple[np.ndarray, str, list[str]]:
    """The time of each chosen frame in ps, where it came from, and notes.

    In this order: ``timestep_fs`` (the MD timestep, times each frame's
    timestep in the file); ``frame_interval_ps`` (the time between
    consecutive frame blocks of the file, times each frame's file position, so
    a block the reader skipped keeps its place); the file's own frame times.
    ValueError when none is available, when both user values are given, or
    when ``timestep_fs`` is given and a chosen frame has no timestep.

    The file's own record of the chosen frames that the axis was not built
    from is held to it (:func:`_file_record_check`): the timesteps, when every
    chosen frame has one, and the printed frame times. A record that does not
    increase, or that puts a frame elsewhere than the axis does, is refused,
    naming the frames, so that a frame missing from the file or a restart
    appended to it is never counted as one more interval; a record that is
    the same in every chosen frame holds no time and is noted. Not checked for
    even spacing here; :func:`collect_tracks` and :func:`collect_bonds` do
    that.
    """
    indices = _frame_list(trajectory, frames)
    if not indices:
        raise ValueError("no frame chosen")
    if timestep_fs is not None and frame_interval_ps is not None:
        raise ValueError("give timestep_fs or frame_interval_ps, not both: "
                         "they are two statements of the same time axis")
    notes: list[str] = []
    file_times = None
    if trajectory.times_ps is not None:
        values = np.array([trajectory.times_ps[k] for k in indices], float)
        if np.isfinite(values).all():
            file_times = values
    constants = _constants()
    if timestep_fs is not None:
        dt_fs = _positive(timestep_fs, "timestep_fs")
        steps = np.array([int(trajectory.timesteps[k]) for k in indices])
        if (steps == NO_TIMESTEP).any():
            missing = [k for k, s in zip(indices, steps) if s == NO_TIMESTEP]
            raise ValueError(
                f"timestep_fs needs the file's timestep of every frame; "
                f"frame(s) {missing[:5]} have none. frame_interval_ps (the time "
                "between frames) gives the axis without them")
        t_ps = steps.astype(np.float64) * dt_fs * (constants.femto
                                                    / constants.pico)
        source = (f"file timesteps x timestep_fs = {dt_fs!r} fs (given by "
                  "the user)")
        if getattr(trajectory, "file_format", "") == "vasp-xdatcar":
            notes.append(
                "the XDATCAR's 'configuration=' numbers are taken as ionic "
                "steps; whether VASP writes the ionic step there or a count "
                "of written configurations when NBLOCK > 1 is not settled "
                "here (reference to verify); frame_interval_ps (POTIM x "
                "NBLOCK) states the axis without that assumption")
    elif frame_interval_ps is not None:
        interval = _positive(frame_interval_ps, "frame_interval_ps")
        positions = np.array([trajectory.file_position(k) for k in indices],
                             dtype=np.float64)
        t_ps = positions * interval
        source = (f"file position x frame_interval_ps = {interval!r} ps "
                  "(given by the user)")
        if trajectory.skipped:
            notes.append("file positions count the blocks the reader skipped, "
                         "so each skipped block keeps its place in time")
    elif file_times is not None:
        t_ps = file_times
        source = "frame times written in the file"
        if trajectory.units_note:
            notes.append(f"frame times as the reader gives them: "
                         f"{trajectory.units_note}")
    else:
        raise ValueError(
            "no time axis: the file gives no frame times, and no timestep is "
            "assumed. timestep_fs (the MD timestep in fs, used with the "
            "file's timesteps) or frame_interval_ps (the time between frames "
            "in ps) states it")
    t_ps = np.asarray(t_ps, dtype=np.float64)
    from_file_times = source == "frame times written in the file"
    axis = ("the file's frame times" if from_file_times else
            "timestep_fs" if timestep_fs is not None else "frame_interval_ps")
    if timestep_fs is None:
        steps = np.array([int(trajectory.timesteps[k]) for k in indices])
        if (steps != NO_TIMESTEP).all():
            # integers, exact; held to an axis with no printing rounding
            # (file positions x interval) at TIME_SPACING_RTOL
            notes += _file_record_check(t_ps, steps, indices, "timesteps",
                                        exact=not from_file_times, axis=axis)
    if file_times is not None and not from_file_times:
        notes += _file_record_check(t_ps, file_times, indices, "frame times",
                                    exact=False, axis=axis)
        span_user = float(t_ps[-1] - t_ps[0])
        span_file = float(file_times[-1] - file_times[0])
        # the 1e-6 only decides whether the disagreement is worth a note;
        # printed times carry about six significant figures
        if not np.allclose(file_times - file_times[0], t_ps - t_ps[0],
                           rtol=1e-6, atol=0.0):
            ratio = (f"; the file's span is {span_file / span_user:.6g} times "
                     "the user's" if span_user > 0 else "")
            notes.append(
                f"the file's own frame times differ from the axis given: the "
                f"chosen frames span {span_file!r} ps in the file and "
                f"{span_user!r} ps from the user's value{ratio}; the user's "
                "value is used")
    return t_ps, source, notes


# ---------------------------------------------------------------------------
# the tracks: continuous positions through time
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AtomTracks:
    """Continuous positions (and velocities) of chosen atoms through time.

    ``unwrapped_cart_ang`` (T, n, 3) and ``vel_ang_per_ps`` (T, n, 3) or None
    hold the atoms in ``rows`` (rows of the trajectory's frames, ascending),
    ``elements``, ``atom_id`` and ``masses_amu`` (gemmi's standard atomic
    weights) one per atom. ``com_ang`` (T, 3) is the centre of mass of the
    whole model, every atom included, whichever atoms are kept. ``t_ps`` are
    the frame times, evenly spaced by ``dt_ps``; ``frames`` the trajectory
    indices. ``box_ang`` (T, 3, 3), ``volume_ang3`` (T,). ``max_step_ang``
    and ``max_step_fraction`` are the largest move of an atom between two
    consecutive frames, in Å and as a fraction of the box width it was
    measured across. Every array is read-only.
    """

    source_path: str
    frames: tuple[int, ...]
    t_ps: np.ndarray
    dt_ps: float
    time_source: str
    elements: np.ndarray
    atom_id: np.ndarray
    rows: np.ndarray
    masses_amu: np.ndarray
    unwrapped_cart_ang: np.ndarray
    vel_ang_per_ps: np.ndarray | None
    com_ang: np.ndarray
    box_ang: np.ndarray
    volume_ang3: np.ndarray
    unwrap_method: str
    max_step_ang: float
    max_step_fraction: float
    n_atoms_model: int
    composition_model: dict[str, int]
    notes: tuple[str, ...] = ()

    @property
    def n_frames(self) -> int:
        return int(self.t_ps.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.elements.shape[0])

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(str(s) for s in np.unique(self.elements))

    @property
    def mean_volume_ang3(self) -> float:
        return math.fsum(self.volume_ang3.tolist()) / self.n_frames

    @property
    def box_varies(self) -> bool:
        return bool((self.box_ang != self.box_ang[0]).any())

    def rows_of(self, element: str) -> np.ndarray:
        """Indices into this object's per-atom arrays of one element's atoms."""
        return np.flatnonzero(self.elements == element)

    @property
    def method_parameters(self) -> dict:
        """How the tracks were made, for ``md_stats.Provenance``."""
        return {"time axis": self.time_source,
                "frame spacing (ps)": self.dt_ps,
                "unwrapped by": self.unwrap_method,
                "largest step (box widths)": self.max_step_fraction}

    def describe(self) -> list[str]:
        lines = [f"source: {self.source_path}",
                 f"{self.n_frames} frames ({self.frames[0]} to "
                 f"{self.frames[-1]}), every {self.dt_ps:.6g} ps, "
                 f"{self.t_ps[0]:.6g} to {self.t_ps[-1]:.6g} ps; times from "
                 f"{self.time_source}",
                 f"{self.n_atoms} of the model's {self.n_atoms_model} atoms: "
                 + ", ".join(f"{s} {int((self.elements == s).sum())}"
                             for s in self.species),
                 f"unwrapped by: {self.unwrap_method}; largest step between "
                 f"frames {self.max_step_ang:.6g} Å, "
                 f"{self.max_step_fraction:.6g} of a box width",
                 "velocities: " + ("from the file" if self.vel_ang_per_ps
                                   is not None else "none in the file")]
        return lines + list(self.notes)


def _step_fraction(delta_cart: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Cartesian steps as fractions of the box vectors (the widths' units)."""
    return np.linalg.solve(box.T, delta_cart.T).T


def _make_tracks(*, source_path, frames, t_ps, time_source, elements, atom_id,
                 rows, positions, velocities, com, boxes, unwrap_method,
                 max_step_ang, max_step_fraction, n_atoms_model,
                 composition_model, notes) -> AtomTracks:
    dt = _even_spacing(t_ps, "tracks")
    masses = _weights_amu(elements)
    volumes = np.abs(np.linalg.det(boxes))
    return AtomTracks(
        source_path=str(source_path), frames=tuple(int(k) for k in frames),
        t_ps=_frozen(np.asarray(t_ps, dtype=np.float64)), dt_ps=dt,
        time_source=time_source, elements=_frozen(elements),
        atom_id=_frozen(np.asarray(atom_id, dtype=np.int64)),
        rows=_frozen(np.asarray(rows, dtype=np.int64)),
        masses_amu=_frozen(masses), unwrapped_cart_ang=_frozen(positions),
        vel_ang_per_ps=None if velocities is None else _frozen(velocities),
        com_ang=_frozen(com), box_ang=_frozen(boxes),
        volume_ang3=_frozen(volumes), unwrap_method=unwrap_method,
        max_step_ang=float(max_step_ang),
        max_step_fraction=float(max_step_fraction),
        n_atoms_model=int(n_atoms_model),
        composition_model=dict(composition_model),
        notes=_unique_notes(notes))


def _element_selection(symbols: np.ndarray, elements, what: str):
    present = sorted({str(s) for s in symbols})
    if elements is None:
        return np.arange(symbols.size), present
    if isinstance(elements, str):
        raise ValueError(f"{what}: elements needs a collection of symbols, not "
                         "one string")
    wanted = sorted({validate_symbol(str(e)) for e in elements})
    if not wanted:
        raise ValueError(f"{what}: no element chosen")
    absent = [e for e in wanted if e not in present]
    if absent:
        raise ValueError(f"{what}: {', '.join(absent)} not in the model "
                         f"(it holds {', '.join(present)})")
    return np.flatnonzero(np.isin(symbols, wanted)), wanted


def collect_tracks(trajectory, *, frames: Sequence[int] | None = None,
                   timestep_fs: float | None = None,
                   frame_interval_ps: float | None = None,
                   elements: Sequence[str] | None = None,
                   unwrap: str = "auto",
                   step_limit_fraction: float = UNWRAP_STEP_LIMIT
                   ) -> AtomTracks:
    """Read the chosen frames once and keep continuous positions through time.

    ``frames``: trajectory indices in increasing order (default every
    readable frame); a frame that cannot be read is refused, not skipped.
    The time axis is :func:`time_axis_ps`'s and has to be evenly spaced.
    ``elements`` keeps only those elements' atoms (memory: 24 bytes per atom
    per frame for the positions, as much again for velocities); the centre
    of mass is still taken over every atom. ``unwrap``: ``'file'`` (the
    file's unwrapped coordinates, refused when a frame has none),
    ``'minimum image'`` (continuity, module docstring), ``'auto'`` (the file's
    when frame 0 carries them). ``step_limit_fraction`` is the largest step
    between two frames accepted, as a fraction of a box width: for the
    continuity unwrap 0 < f <= 0.5; for the file's unwrapped coordinates any
    f > 0, a value above 0.5 stating that the file's coordinates are
    continuous across such steps. A larger step is refused, naming the atom
    and the frames: in the file's unwrapped coordinates it is what image
    flags reset between two frames leave behind.
    """
    _tracking_check(trajectory)
    method = _choice(unwrap, "unwrap", UNWRAP_METHODS)
    limit = _positive(step_limit_fraction, "step_limit_fraction")
    indices = _frame_list(trajectory, frames)
    if len(indices) < 2:
        raise ValueError("dynamics needs at least 2 frames")
    t_ps, time_source, notes = time_axis_ps(
        trajectory, indices, timestep_fs=timestep_fs,
        frame_interval_ps=frame_interval_ps)
    _even_spacing(t_ps, f"frames {indices[0]} to {indices[-1]}", indices)
    notes += [f"reader: {n}" for n in trajectory.notes]
    if trajectory.units_note and not any(trajectory.units_note in n
                                         for n in notes):
        notes.append(f"units as the reader gives them (positions, times and "
                     f"velocities): {trajectory.units_note}")

    first = _load(trajectory, indices[0])
    rows, kept = _element_selection(first.elements, elements, "collect_tracks")
    if method == "auto":
        method = "file" if first.unwrapped_cart_ang is not None else \
            "minimum image"
    if method == "file" and first.unwrapped_cart_ang is None:
        raise ValueError(f"frame {indices[0]} carries no unwrapped "
                         "coordinates; unwrap='minimum image' unwraps by "
                         "continuity instead")
    if method == "minimum image" and limit > 0.5:
        raise ValueError(f"step_limit_fraction is {limit}; above 0.5 a step "
                         "and its periodic image cannot be told apart by "
                         "continuity")
    n_frames, n_model, n_kept = len(indices), first.n_atoms, rows.size
    weights_all = _weights_amu(first.elements)
    total_mass = math.fsum(weights_all.tolist())
    positions = np.empty((n_frames, n_kept, 3))
    velocities = np.empty((n_frames, n_kept, 3)) \
        if first.vel_ang_per_ps is not None else None
    boxes = np.empty((n_frames, 3, 3))
    com = np.empty((n_frames, 3))
    max_fraction, max_step = 0.0, 0.0
    worst = None
    previous = None
    unwrapped = None
    for slot, k in enumerate(indices):
        frame = first if slot == 0 else _load(trajectory, k)
        if method == "file":
            if frame.unwrapped_cart_ang is None:
                raise ValueError(
                    f"frame {k} carries no unwrapped coordinates, though frame "
                    f"{indices[0]} does; unwrap='minimum image' unwraps every "
                    "frame by continuity instead")
            current = np.asarray(frame.unwrapped_cart_ang, dtype=np.float64)
            if previous is not None:
                step = current - unwrapped
                largest = np.abs(_step_fraction(step, frame.box_ang)).max(
                    axis=1)
                lengths = np.linalg.norm(step, axis=1)
                atom = int(np.argmax(largest))
                if largest[atom] > max_fraction:
                    max_fraction = float(largest[atom])
                    worst = (atom, indices[slot - 1], k)
                max_step = max(max_step, float(lengths.max()))
                if max_fraction > limit:
                    a, k0, k1 = worst
                    raise ValueError(
                        f"atom id {int(frame.atom_id[a])} ({frame.elements[a]}) "
                        f"moves {max_fraction:.4g} of a box width between "
                        f"frames {k0} and {k1} in the file's unwrapped "
                        f"coordinates, beyond step_limit_fraction = "
                        f"{limit:g}: a jump of that size between two frames "
                        "is what image flags reset or rewritten between them "
                        "leave behind (a restart from a file without image "
                        "flags, a post-processed dump). unwrap='minimum image' "
                        "unwraps by continuity and does not use the image "
                        "flags; a larger step_limit_fraction, given "
                        "explicitly, takes the file's coordinates as "
                        "continuous across such steps")
            unwrapped = current
        else:
            if previous is None:
                unwrapped = np.array(frame.cart_ang, dtype=np.float64)
            else:
                same = np.array_equal(frame.box_ang, previous.box_ang) and \
                    np.array_equal(frame.origin_ang, previous.origin_ang)
                if same:
                    fraction = frame.frac - previous.frac
                else:
                    fraction = _step_fraction(frame.cart_ang - previous.cart_ang,
                                              frame.box_ang)
                fraction = fraction - np.rint(fraction)
                step = fraction @ frame.box_ang
                largest = np.abs(fraction).max(axis=1)
                atom = int(np.argmax(largest))
                if largest[atom] > max_fraction:
                    max_fraction = float(largest[atom])
                    worst = (atom, indices[slot - 1], k)
                max_step = max(max_step,
                               float(np.linalg.norm(step, axis=1).max()))
                if max_fraction > limit:
                    a, k0, k1 = worst
                    raise ValueError(
                        f"atom id {int(frame.atom_id[a])} ({frame.elements[a]}) "
                        f"moves {max_fraction:.4g} of a box width between "
                        f"frames {k0} and {k1}, beyond step_limit_fraction = "
                        f"{limit:g}: a step of more than half the width "
                        "cannot be told from a shorter one the other way, so "
                        "the unwrap by continuity is refused. Frames closer in "
                        "time, unwrapped coordinates in the file (xu yu zu, or "
                        "image flags ix iy iz), or a larger step_limit_fraction "
                        "given explicitly (at most 0.5) resolve it")
                unwrapped = unwrapped + step
        positions[slot] = unwrapped[rows]
        if velocities is not None:
            if frame.vel_ang_per_ps is None:
                velocities = None
                notes.append(f"frame {k} carries no velocities, though frame "
                             f"{indices[0]} does; no velocities are kept")
            else:
                velocities[slot] = frame.vel_ang_per_ps[rows]
        boxes[slot] = frame.box_ang
        com[slot] = (weights_all @ unwrapped) / total_mass
        previous = frame

    span = (trajectory.file_position(indices[0]),
            trajectory.file_position(indices[-1]))
    inside = [p for p in trajectory.skipped if span[0] < p < span[1]]
    if inside:
        notes.append(f"the reader skipped file block(s) {inside[:5]} between "
                     "the chosen frames; the time axis was checked for the "
                     "gap they leave")
    notes.append(f"unwrapped by: {method}" + (
        " (the file's unwrapped coordinates)" if method == "file" else
        " (each step taken as the shortest the periodic box allows)"))
    notes.append(f"largest step between consecutive frames: {max_step:.6g} Å, "
                 f"{max_fraction:.6g} of a box width (step_limit_fraction = "
                 f"{limit:g}, the largest accepted)")
    notes.append("centre of mass over all " + str(n_model) + " atoms; "
                 + _masses_note(first.elements))
    if np.any(boxes != boxes[0]):
        notes.append("the box changes between frames")
        if method == "file":
            notes.append(
                "in a box that changes, the file's unwrapped coordinates are "
                "unwrapped by image counts (LAMMPS's xu = x + ix Lx), which "
                "Bullerjahn et al. (2023) show lets the box rescaling enter "
                "the displacements and the diffusion coefficient; "
                "unwrap='minimum image' gives their recommended TOR scheme")
    return _make_tracks(
        source_path=trajectory.source_path, frames=indices, t_ps=t_ps,
        time_source=time_source, elements=first.elements[rows],
        atom_id=first.atom_id[rows], rows=rows, positions=positions,
        velocities=velocities, com=com, boxes=boxes, unwrap_method=method,
        max_step_ang=max_step, max_step_fraction=max_fraction,
        n_atoms_model=n_model, composition_model=first.composition,
        notes=notes)


def tracks_from_arrays(elements, unwrapped_cart_ang, *, t_ps, box_ang,
                       vel_ang_per_ps=None, atom_id=None,
                       source_path: str = "<memory>",
                       notes: Sequence[str] = (),
                       step_limit_fraction: float = UNWRAP_STEP_LIMIT
                       ) -> AtomTracks:
    """Tracks from arrays already in memory (synthetic models, other tools).

    ``unwrapped_cart_ang`` (T, N, 3) continuous positions of every atom of the
    model; ``t_ps`` (T,) evenly spaced; ``box_ang`` (3, 3) rows a, b, c, or
    (T, 3, 3). The centre of mass is over these N atoms.

    Positions wrapped into the box (the default of several MD toolkits) jump
    by about a box width when an atom crosses a face, and give an MSD that
    has no meaning; they cannot be told from continuous ones except by the
    size of the steps. A step between consecutive frames larger than
    ``step_limit_fraction`` of a box width is therefore refused, naming the
    atom and the frames; a value above 0.5, given explicitly, states that the
    positions are continuous across such steps.
    """
    limit = _positive(step_limit_fraction, "step_limit_fraction")
    if isinstance(notes, (str, bytes)):
        raise ValueError("notes needs a sequence of sentences, not one string")
    if isinstance(elements, (str, bytes)):
        raise ValueError("elements needs one symbol per atom, not one string")
    symbols = np.asarray([validate_symbol(str(s)) for s in elements],
                         dtype="<U2")
    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("elements needs one symbol per atom")
    positions = np.asarray(unwrapped_cart_ang, dtype=np.float64)
    n = symbols.size
    if positions.ndim != 3 or positions.shape[1:] != (n, 3):
        raise ValueError(f"unwrapped_cart_ang has shape {positions.shape}; "
                         f"(n_frames, {n}, 3) is needed")
    if not np.isfinite(positions).all():
        raise ValueError("unwrapped_cart_ang holds a NaN or infinite value")
    times = np.asarray(t_ps, dtype=np.float64)
    if times.shape != (positions.shape[0],):
        raise ValueError(f"t_ps needs {positions.shape[0]} values, one per "
                         "frame")
    box = np.asarray(box_ang, dtype=np.float64)
    if box.shape == (3, 3):
        boxes = np.repeat(_checked_box(box)[None], positions.shape[0], axis=0)
    elif box.shape == (positions.shape[0], 3, 3):
        boxes = np.array([_checked_box(b) for b in box])
    else:
        raise ValueError(f"box_ang has shape {box.shape}; (3, 3) or "
                         f"({positions.shape[0]}, 3, 3) is needed")
    velocities = None
    if vel_ang_per_ps is not None:
        velocities = np.asarray(vel_ang_per_ps, dtype=np.float64)
        if velocities.shape != positions.shape:
            raise ValueError(f"vel_ang_per_ps has shape {velocities.shape}; "
                             f"{positions.shape} is needed")
        if not np.isfinite(velocities).all():
            raise ValueError("vel_ang_per_ps holds a NaN or infinite value")
    ids = np.arange(n) if atom_id is None else np.asarray(atom_id)
    if ids.shape != (n,):
        raise ValueError(f"atom_id needs {n} values")
    weights = _weights_amu(symbols)
    com = np.einsum("tna,n->ta", positions, weights) / math.fsum(
        weights.tolist())
    _even_spacing(times, "tracks", range(times.size))
    # steps as fractions of the box vectors, d @ inv(box), a chunk of frames
    # at a time (about twice as fast as one solve per frame, measured)
    inverse = np.linalg.inv(boxes)
    chunk = max(1, CHUNK_VALUES // (3 * n))
    max_fraction, max_step = 0.0, 0.0
    for start in range(1, positions.shape[0], chunk):
        stop = min(positions.shape[0], start + chunk)
        step = positions[start:stop] - positions[start - 1:stop - 1]
        largest = np.abs(np.matmul(step, inverse[start:stop])).max(axis=2)
        per_frame = largest.max(axis=1)
        max_step = max(max_step, float(np.sqrt(np.einsum(
            "tna,tna->tn", step, step).max())))
        beyond = np.flatnonzero(per_frame > limit)
        if beyond.size:
            row = int(beyond[0])
            k = start + row - 1
            atom = int(np.argmax(largest[row]))
            raise ValueError(
                f"atom {atom} (id {ids[atom]}, {symbols[atom]}) moves "
                f"{float(largest[row, atom]):.4g} of a box width between "
                f"frames {k} and {k + 1}, beyond step_limit_fraction = "
                f"{limit:g}: positions wrapped into the box jump by about a "
                "box width when an atom crosses a face, and "
                "unwrapped_cart_ang needs continuous positions. Unwrapping "
                "them first (or collect_tracks on the file), or a larger "
                "step_limit_fraction given explicitly for positions known to "
                "be continuous, resolves it")
        max_fraction = max(max_fraction, float(per_frame.max()))
    composition = {str(s): int(c) for s, c in
                   zip(*np.unique(symbols, return_counts=True))}
    return _make_tracks(
        source_path=source_path, frames=range(positions.shape[0]),
        t_ps=times, time_source="t_ps given with the arrays",
        elements=symbols, atom_id=ids, rows=np.arange(n), positions=positions,
        velocities=velocities, com=com, boxes=boxes,
        unwrap_method="given as continuous positions",
        max_step_ang=max_step, max_step_fraction=max_fraction,
        n_atoms_model=n, composition_model=composition,
        notes=list(notes) + [
            f"largest step between consecutive frames: {max_step:.6g} Å, "
            f"{max_fraction:.6g} of a box width (step_limit_fraction = "
            f"{limit:g}, the largest accepted)",
            "centre of mass over all " + str(n) + " atoms; "
            + _masses_note(symbols)])


def _check_tracks(tracks) -> AtomTracks:
    if not isinstance(tracks, AtomTracks):
        raise ValueError(f"an AtomTracks is needed (collect_tracks or "
                         f"tracks_from_arrays builds one), not "
                         f"{type(tracks).__name__}")
    return tracks


def _species(tracks: AtomTracks, elements) -> list[str]:
    if elements is None:
        return list(tracks.species)
    if isinstance(elements, str):
        raise ValueError("elements needs a collection of symbols, not one "
                         "string")
    wanted = [validate_symbol(str(e)) for e in elements]
    absent = [e for e in wanted if e not in tracks.species]
    if absent:
        raise ValueError(f"{', '.join(absent)} not in the tracks (they hold "
                         f"{', '.join(tracks.species)})")
    return sorted(set(wanted))


def _blocks(n_frames: int, n_blocks, labels: Sequence[int], what: str
            ) -> tuple[list[tuple[int, int]], int, list[str]]:
    count = _whole(n_blocks, "n_blocks", 1)
    length = n_frames // count
    if length < 2:
        raise ValueError(f"{what}: {n_frames} frames cut into {count} blocks "
                         f"leave {length} frame(s) per block; a block needs at "
                         "least 2")
    bounds = [(b * length, (b + 1) * length) for b in range(count)]
    notes = []
    unused = n_frames - count * length
    if unused:
        notes.append(f"{what}: the last {unused} frame(s) (trajectory frames "
                     f"{labels[n_frames - unused]} to {labels[-1]}) fall in no "
                     f"block: {n_frames} frames make {count} blocks of "
                     f"{length}")
    if count == 1:
        notes.append(f"{what}: one block of {length} frames, every time "
                     "origin in it; n_blocks > 1 gives a spread across blocks")
    else:
        notes.append(f"{what}: {count} blocks of {length} frames, every time "
                     "origin inside each block; std is the spread across "
                     "blocks (ddof = 1)")
    return bounds, length, notes


def _lag_count(max_lag_t_ps, dt: float, length: int
               ) -> tuple[int, list[str]]:
    """How many lags (0 .. m) reach ``max_lag_t_ps``, and a note when it was
    taken to the nearest whole frame."""
    if max_lag_t_ps is None:
        return length, []
    t = _positive(max_lag_t_ps, "max_lag_t_ps")
    m = int(round(t / dt))
    if m < 1:
        raise ValueError(f"max_lag_t_ps {t} ps is less than one frame "
                         f"({dt:.6g} ps)")
    if m > length - 1:
        raise ValueError(f"max_lag_t_ps {t} ps is {m} frames; a block of "
                         f"{length} frames reaches {length - 1} frames "
                         f"({(length - 1) * dt:.6g} ps)")
    notes = []
    if abs(m * dt - t) > 1e-9 * dt:
        notes.append(f"max_lag_t_ps taken to the nearest whole frame: "
                     f"{t:.6g} -> {m * dt:.6g} ps ({m} frame(s))")
    return m + 1, notes


def _lags(lag_t_ps, dt: float, length: int, minimum: int
          ) -> tuple[np.ndarray, list[str]]:
    values = np.atleast_1d(np.asarray(lag_t_ps, dtype=np.float64))
    if values.ndim != 1 or values.size == 0:
        raise ValueError("lag_t_ps needs one or more lag times in ps")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("lag_t_ps holds a value that is negative, NaN or "
                         "infinite")
    frames = np.rint(values / dt).astype(np.int64)
    notes = []
    for value, m in zip(values.tolist(), frames.tolist()):
        if m < minimum:
            raise ValueError(
                f"lag {value!r} ps is {m} frames; the smallest lag here is "
                f"{minimum} frame(s) ({minimum * dt:.6g} ps)")
        if m > length - 1:
            raise ValueError(
                f"lag {value!r} ps is {m} frames; a block of {length} frames "
                f"reaches {length - 1} ({(length - 1) * dt:.6g} ps)")
    moved = np.abs(frames * dt - values) > 1e-9 * dt
    if moved.any():
        notes.append("lag times taken to the nearest whole frame: " + ", ".join(
            f"{v:.6g} -> {m * dt:.6g} ps" for v, m in
            zip(values[moved], frames[moved])))
    unique = np.unique(frames)
    if unique.size < frames.size:
        notes.append(f"{frames.size - unique.size} lag time(s) fell on the "
                     "same frame as another and were merged")
    return unique, notes


def log_spaced_lag_times(tracks: AtomTracks, n_points: int, *,
                         n_blocks: int = 1) -> np.ndarray:
    """About ``n_points`` lag times from one frame to the end of a block,
    evenly spaced in log t (whole frames, repeats merged), in ps."""
    tracks = _check_tracks(tracks)
    count = _whole(n_points, "n_points", 1)
    _, length, _ = _blocks(tracks.n_frames, n_blocks, tracks.frames, "lags")
    if length < 2:
        raise ValueError("a block of one frame has no lag")
    frames = np.unique(np.rint(np.geomspace(1, length - 1, count)).astype(int))
    return frames * tracks.dt_ps


def _block_labels(n_blocks: int) -> list[int]:
    return list(range(n_blocks))


def _block_frames(tracks: AtomTracks, bounds) -> tuple[tuple[int, int], ...]:
    return tuple((tracks.frames[s], tracks.frames[e - 1]) for s, e in bounds)


def _block_spans(blocks) -> str:
    """'frames 0-99, 100-199' for a result's ``blocks``, for notes."""
    if not blocks:
        return "frames not recorded"
    return "frames " + ", ".join(f"{int(a)}-{int(b)}" for a, b in blocks)


# ---------------------------------------------------------------------------
# correlation sums by FFT
# ---------------------------------------------------------------------------

def _atoms_first(vectors: np.ndarray, rows: np.ndarray,
                 shift: np.ndarray | None) -> np.ndarray:
    """(c, 3, L) C-contiguous copy of the chosen atoms of ``vectors``
    (L, N, 3), time last, so that every FFT runs along contiguous memory
    (25-30 % faster than along the first axis on 10 648 atoms x 1 000
    frames, measured)."""
    part = vectors[:, rows, :]
    if shift is not None:
        part = part - shift[:, None, :]
    return np.ascontiguousarray(part.transpose(1, 2, 0))


def _power_sum(part: np.ndarray, nfft: int, weights: np.ndarray | None
               ) -> np.ndarray:
    """sum over atoms (weighted) and axes of |rfft|^2 of part (c, 3, L)."""
    spectrum = _fft().rfft(part, n=nfft, axis=-1)
    flat = spectrum.view(np.float64)
    np.multiply(flat, flat, out=flat)
    pairs = flat.reshape(part.shape[0], 3, -1, 2)
    if weights is None:
        return pairs.sum(axis=(0, 1, 3))
    return np.einsum("cakr,c->k", pairs, weights)


def _msd_sums(vectors: np.ndarray, n_lags: int, *,
              rows: np.ndarray | None = None,
              shift: np.ndarray | None = None,
              weights: np.ndarray | None = None) -> np.ndarray:
    """sum_i w_i MSD_i(m), m = 0 .. n_lags - 1, every origin.

    ``vectors`` (L, N, 3), positions in Å or charge displacements in e Å;
    ``rows`` the atoms to use (default all), copied one chunk at a time;
    ``shift`` (L, 3) is subtracted from every position first.
    """
    sfft = _fft()
    length = vectors.shape[0]
    rows = np.arange(vectors.shape[1]) if rows is None else rows
    nfft = sfft.next_fast_len(2 * length - 1, real=True)
    lags = np.arange(n_lags)
    total = np.zeros(n_lags)
    chunk = max(1, CHUNK_VALUES // (3 * nfft))
    for start in range(0, rows.size, chunk):
        part = _atoms_first(vectors, rows[start:start + chunk], shift)
        part -= part.mean(axis=-1, keepdims=True)
        w = None if weights is None else weights[start:start + chunk]
        squares = np.einsum("cal,cal->cl", part, part)
        squares = squares.sum(axis=0) if w is None else w @ squares
        prefix = np.concatenate(([0.0], np.cumsum(squares)))
        s1 = prefix[length - lags] + (prefix[length] - prefix[lags])
        acf = sfft.irfft(_power_sum(part, nfft, w), n=nfft)[:n_lags]
        total += s1 - 2.0 * acf
    out = total / (length - lags)
    out[0] = 0.0
    return out


def _acf_sums(vel_ang_per_ps: np.ndarray, n_lags: int, *,
              rows: np.ndarray | None = None,
              weights: np.ndarray | None = None) -> np.ndarray:
    """sum_i w_i <v_i(k) . v_i(k + m)>_k, every origin; ``vel_ang_per_ps``
    (L, N, 3), ``rows`` the atoms to use (default all), ``weights`` one per
    row used."""
    sfft = _fft()
    length = vel_ang_per_ps.shape[0]
    rows = np.arange(vel_ang_per_ps.shape[1]) if rows is None else rows
    nfft = sfft.next_fast_len(2 * length - 1, real=True)
    lags = np.arange(n_lags)
    total = np.zeros(n_lags)
    chunk = max(1, CHUNK_VALUES // (3 * nfft))
    for start in range(0, rows.size, chunk):
        part = _atoms_first(vel_ang_per_ps, rows[start:start + chunk], None)
        w = None if weights is None else weights[start:start + chunk]
        total += sfft.irfft(_power_sum(part, nfft, w), n=nfft)[:n_lags]
    return total / (length - lags)


def _rows_of(containers) -> list[dict]:
    rows: list[dict] = []
    for container in containers:
        rows.extend(container.as_rows())
    return rows


# ---------------------------------------------------------------------------
# mean-square displacement and diffusion
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MSDResult:
    """MSD per element (Å^2) on the lag axis, one row per block.

    ``com_msd`` is the MSD of the model's centre of mass over the same lags,
    measured whether or not it was removed. ``blocks`` gives the first and
    last trajectory frame of each block.
    """

    msd: dict[str, Series]
    com_msd: Series
    n_atoms: dict[str, int]
    dt_ps: float
    block_length: int
    blocks: tuple[tuple[int, int], ...]
    remove_com_drift: bool
    notes: tuple[str, ...] = ()

    @property
    def lag_t_ps(self) -> np.ndarray:
        return next(iter(self.msd.values())).axis

    @property
    def method_parameters(self) -> dict:
        return {"n_blocks": len(self.blocks),
                "frames per block": self.block_length,
                "MSD time origins": "every frame of the block",
                "centre-of-mass drift removed": self.remove_com_drift}

    def as_rows(self) -> list[dict]:
        return _rows_of(list(self.msd.values()) + [self.com_msd])


def msd(tracks: AtomTracks, *, remove_com_drift: bool = False,
        n_blocks: int = 1, max_lag_t_ps: float | None = None,
        elements: Sequence[str] | None = None) -> MSDResult:
    """Mean-square displacement per element, every time origin, by FFT.

    ``remove_com_drift``: subtract the model's centre-of-mass position from
    every atom first (stated in the result either way). ``max_lag_t_ps``
    (default: the whole block) ends the lag axis.
    """
    tracks = _check_tracks(tracks)
    if not isinstance(remove_com_drift, (bool, np.bool_)):
        raise ValueError("remove_com_drift is True or False")
    species = _species(tracks, elements)
    bounds, length, notes = _blocks(tracks.n_frames, n_blocks, tracks.frames,
                                    "MSD")
    n_lags, lag_notes = _lag_count(max_lag_t_ps, tracks.dt_ps, length)
    notes += lag_notes
    lag_t = np.arange(n_lags) * tracks.dt_ps
    rows = {el: tracks.rows_of(el) for el in species}
    per = {el: np.empty((len(bounds), n_lags)) for el in species}
    com_rows = np.empty((len(bounds), n_lags))
    for b, (start, stop) in enumerate(bounds):
        positions = tracks.unwrapped_cart_ang[start:stop]
        com = tracks.com_ang[start:stop]
        for el in species:
            per[el][b] = _msd_sums(
                positions, n_lags, rows=rows[el],
                shift=com if remove_com_drift else None) / rows[el].size
        com_rows[b] = _msd_sums(com[:, None, :], n_lags)
    notes.append("MSD by the FFT algorithm (nMOLDYN), every time origin of "
                 "each block; at lag m a block of L frames holds L - m "
                 "origins, one at the last lag")
    notes.append("centre-of-mass drift " + (
        "removed: the model's centre of mass (every atom, standard atomic "
        "weights) is subtracted from each position" if remove_com_drift else
        "not removed; com_msd gives the centre of mass's own MSD"))
    if not remove_com_drift:
        last = com_rows[:, -1].mean()
        parts = [f"{el} {last / per[el][:, -1].mean():.4g}"
                 for el in species if per[el][:, -1].mean() > 0]
        notes.append(
            f"at the last lag ({lag_t[-1]:.6g} ps) the centre of mass's own "
            f"MSD is {last:.6g} Å^2 (block mean)"
            + ("; as a fraction of each element's MSD there: "
               + ", ".join(parts) if parts else ""))
    labels = _block_labels(len(bounds))
    series = {el: Series(f"MSD {el}", lag_t, "t_ps", "ps", "Å^2", per[el],
                         frames=labels, row_kind="block")
              for el in species}
    com_series = Series("MSD of the centre of mass", lag_t, "t_ps", "ps",
                        "Å^2", com_rows, frames=labels, row_kind="block")
    return MSDResult(msd=series, com_msd=com_series,
                     n_atoms={el: int(rows[el].size) for el in species},
                     dt_ps=tracks.dt_ps, block_length=length,
                     blocks=_block_frames(tracks, bounds),
                     remove_com_drift=bool(remove_com_drift),
                     notes=_unique_notes(list(tracks.notes) + notes))


def _line_fit(t: np.ndarray, y: np.ndarray):
    t_mean, y_mean = t.mean(), y.mean()
    centred = t - t_mean
    slope = float((centred * (y - y_mean)).sum() / (centred * centred).sum())
    intercept = float(y_mean - slope * t_mean)
    return slope, intercept, y - (intercept + slope * t)


def _window(axis: np.ndarray, t_min_ps, t_max_ps, dt: float) -> np.ndarray:
    low = _positive(t_min_ps, "t_min_ps", allow_zero=True)
    high = _positive(t_max_ps, "t_max_ps")
    if high <= low:
        raise ValueError(f"t_max_ps {high} is not above t_min_ps {low}")
    # a rounding allowance, so that a window end typed as 0.3 takes the lag
    # at 3 x 0.1 ps, which floating point puts at 0.30000000000000004
    slack = 1e-9 * dt
    if high > axis[-1] + slack:
        raise ValueError(f"t_max_ps {high} ps lies beyond the last lag, "
                         f"{axis[-1]:.6g} ps")
    chosen = np.flatnonzero((axis >= low - slack) & (axis <= high + slack))
    if chosen.size < 2:
        raise ValueError(f"the window {low:g} to {high:g} ps holds "
                         f"{chosen.size} lag(s); a straight line needs at "
                         "least 2")
    return chosen


def _window_fit(series: Series, t_min_ps, t_max_ps, dt: float) -> dict:
    axis = series.axis
    chosen = _window(axis, t_min_ps, t_max_ps, dt)
    t = axis[chosen]
    y = series.mean[chosen]
    notes = []
    if not np.isfinite(y).all():
        raise ValueError(f"{series.name}: a value in the window is NaN")
    slope, intercept, residual = _line_fit(t, y)
    if chosen.size == 2:
        notes.append(f"{series.name}: 2 lags in the window, so the line passes "
                     "through both and the residual is 0 by construction")
    positive = (t > 0) & (y > 0)
    if positive.sum() >= 2:
        loglog = _line_fit(np.log(t[positive]), np.log(y[positive]))[0]
    else:
        loglog = float("nan")
        notes.append(f"{series.name}: fewer than 2 lags in the window with "
                     "t > 0 and a value above 0, so no log-log slope")
    if positive.sum() < (t > 0).sum():
        notes.append(f"{series.name}: {int((t > 0).sum() - positive.sum())} "
                     "lag(s) in the window have a value of 0 or below and are "
                     "left out of the log-log slope")
    per_block = np.array([_line_fit(t, row[chosen])[0]
                          for row in series.per_frame])
    return {"t": t, "slope": slope, "intercept": intercept,
            "rms": float(np.sqrt(np.mean(residual * residual))),
            "max_abs": float(np.abs(residual).max()), "loglog": float(loglog),
            "per_block_slope": per_block, "notes": notes,
            "labels": [int(v) for v in series.frames]}


@dataclass(frozen=True, eq=False)
class DiffusionFit:
    """D = slope / 6 of a line fitted to the block-mean MSD over a window.

    ``t_min_ps`` / ``t_max_ps`` are the window asked for; ``t_first_ps`` /
    ``t_last_ps`` the first and last lag in it, ``n_points`` how many. The
    residuals are of the block-mean MSD about the line. ``loglog_slope`` is
    the slope of ln MSD against ln t over the window. ``per_block`` holds D
    fitted to each block's MSD over the same window (Å^2/ps).
    ``com_d_ang2_per_ps`` is slope / 6 of the centre of mass's own MSD over
    the same lags when the drift was not removed (None when it was): the
    part of a drifting model's D that the whole model's motion accounts for.
    ``blocks`` gives the first and last trajectory frame of each block of
    ``per_block`` (the MSD's), so that per-block values from two results are
    paired only when they cover the same frames.
    """

    element: str
    t_min_ps: float
    t_max_ps: float
    t_first_ps: float
    t_last_ps: float
    n_points: int
    slope_ang2_per_ps: float
    intercept_ang2: float
    d_ang2_per_ps: float
    d_m2_per_s: float
    rms_residual_ang2: float
    max_abs_residual_ang2: float
    loglog_slope: float
    per_block: Scalar
    com_d_ang2_per_ps: float | None = None
    blocks: tuple[tuple[int, int], ...] = ()
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        return [{"descriptor": f"diffusion coefficient {self.element}",
                 "D (Å^2/ps)": self.d_ang2_per_ps,
                 "D (m^2/s)": self.d_m2_per_s,
                 "std over blocks (Å^2/ps)": self.per_block.std,
                 "blocks": self.per_block.n_frames,
                 "fit window (ps)": f"{self.t_min_ps!r} to {self.t_max_ps!r}",
                 "lags fitted": self.n_points,
                 "slope (Å^2/ps)": self.slope_ang2_per_ps,
                 "intercept (Å^2)": self.intercept_ang2,
                 "rms residual (Å^2)": self.rms_residual_ang2,
                 "log-log slope": self.loglog_slope,
                 "centre of mass slope / 6 (Å^2/ps)": self.com_d_ang2_per_ps}]


def fit_diffusion(msd_result: MSDResult, t_min_ps: float, t_max_ps: float,
                  *, elements: Sequence[str] | None = None
                  ) -> dict[str, DiffusionFit]:
    """D per element from the MSD over ``t_min_ps`` <= t <= ``t_max_ps``.

    The window is required: where the ballistic and caged regimes end and the
    diffusive one begins is a property of the system, read off the MSD (the
    log-log slope in the result helps), never assumed here.
    """
    if not isinstance(msd_result, MSDResult):
        raise ValueError(f"an MSDResult is needed, not "
                         f"{type(msd_result).__name__}")
    if isinstance(elements, str):
        raise ValueError("elements needs a collection of symbols, not one "
                         "string")
    constants = _constants()
    to_si = constants.angstrom ** 2 / constants.pico
    wanted = list(msd_result.msd) if elements is None else \
        [validate_symbol(str(e)) for e in elements]
    com_d_ang2_per_ps = None
    if not msd_result.remove_com_drift:
        com_fit = _window_fit(msd_result.com_msd, t_min_ps, t_max_ps,
                              msd_result.dt_ps)
        com_d_ang2_per_ps = com_fit["slope"] / 6.0
    out = {}
    for el in wanted:
        if el not in msd_result.msd:
            raise ValueError(f"no MSD for {el}; the result holds "
                             f"{', '.join(msd_result.msd)}")
        series = msd_result.msd[el]
        fit = _window_fit(series, t_min_ps, t_max_ps, msd_result.dt_ps)
        d_ang2_per_ps = fit["slope"] / 6.0
        notes = [
            f"D = slope / 6 of a least-squares line through the block-mean "
            f"MSD of {el} at {fit['t'].size} lags from {fit['t'][0]:.6g} to "
            f"{fit['t'][-1]:.6g} ps (the window is the user's choice); MSD "
            "values at neighbouring lags are correlated, so no standard error "
            "of the slope is given; per_block holds D fitted to each block",
            f"log-log slope of the MSD over the window: {fit['loglog']:.6g} "
            "(1 for diffusive motion, 2 for ballistic)"] + fit["notes"]
        if com_d_ang2_per_ps is None:
            notes.append("centre-of-mass drift removed before the MSD")
        else:
            share = (f", {com_d_ang2_per_ps / d_ang2_per_ps:.4g} of this D"
                     if d_ang2_per_ps != 0 else "")
            notes.append(
                f"centre-of-mass drift not removed: the model's centre of "
                f"mass, fitted over the same lags, gives slope / 6 = "
                f"{com_d_ang2_per_ps:.6g} Å^2/ps" + share
                + "; a motion of the whole model enters every element's MSD, "
                "and msd(..., remove_com_drift=True) subtracts it")
        if d_ang2_per_ps < 0:
            notes.append(f"the fitted slope is {fit['slope']:.6g} Å^2/ps, "
                         "below 0: the MSD falls across the window, and the "
                         "value is not a diffusion coefficient")
        per_block = Scalar(f"D {el}", "Å^2/ps", fit["per_block_slope"] / 6.0,
                           frames=fit["labels"], row_kind="block")
        out[el] = DiffusionFit(
            element=el, t_min_ps=float(t_min_ps), t_max_ps=float(t_max_ps),
            t_first_ps=float(fit["t"][0]), t_last_ps=float(fit["t"][-1]),
            n_points=int(fit["t"].size), slope_ang2_per_ps=fit["slope"],
            intercept_ang2=fit["intercept"], d_ang2_per_ps=d_ang2_per_ps,
            d_m2_per_s=d_ang2_per_ps * to_si, rms_residual_ang2=fit["rms"],
            max_abs_residual_ang2=fit["max_abs"], loglog_slope=fit["loglog"],
            per_block=per_block,
            com_d_ang2_per_ps=None if com_d_ang2_per_ps is None
            else float(com_d_ang2_per_ps),
            blocks=tuple(msd_result.blocks),
            notes=_unique_notes(notes + list(msd_result.notes)))
    return out


# ---------------------------------------------------------------------------
# self correlations at chosen lags: alpha2, G_s, F_s
# ---------------------------------------------------------------------------

def _edges(edges_r_ang) -> np.ndarray:
    edges = np.asarray(edges_r_ang, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges_r_ang needs at least 2 bin edges")
    if not np.isfinite(edges).all() or edges[0] < 0:
        raise ValueError("edges_r_ang holds a NaN, infinite or negative edge")
    if not (np.diff(edges) > 0).all():
        raise ValueError("edges_r_ang has to increase strictly")
    return edges


def _binner(edges: np.ndarray):
    """numpy.histogram on these edges. Edges that are exactly
    ``numpy.linspace(first, last, n + 1)`` go through numpy's even-bin path,
    which places each value against those same edges (so the counts are
    identical) and was about 30 % faster on 1e7 values (measured)."""
    even = np.array_equal(edges, np.linspace(edges[0], edges[-1], edges.size))
    if even:
        span = (float(edges[0]), float(edges[-1]))
        bins = edges.size - 1
        return lambda values: np.histogram(values, bins=bins, range=span)[0]
    return lambda values: np.histogram(values, bins=edges)[0]


def _sin_ratio_sum(r_ang: np.ndarray, q_inv_ang: float) -> float:
    """sum of sin(q r) / (q r), with the limit 1 at r = 0."""
    qr = q_inv_ang * r_ang
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(qr > 0.0, np.sin(qr) / qr, 1.0)
    return float(values.sum())


def _q_values(q_inv_ang) -> np.ndarray:
    q = np.atleast_1d(np.asarray(q_inv_ang, dtype=np.float64))
    if q.ndim != 1 or q.size == 0:
        raise ValueError("q_inv_ang needs one or more q values in 1/Å")
    if not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError("q_inv_ang holds a value that is not above 0, or is "
                         "not finite")
    return q


@dataclass(frozen=True, eq=False)
class SelfCorrelations:
    """Displacement statistics at the chosen lags, per element, rows = blocks.

    ``msd`` (Å^2) and ``msd4`` (Å^4) are <dr^2> and <dr^4> computed directly,
    ``alpha2`` the non-Gaussian parameter. ``van_hove[el][i]`` is the
    Histogram of |dr| at lag i as a density (1/Å; 4 pi r^2 G_s) and
    ``gs_inv_ang3[el][i]`` G_s itself on the bin centres; ``isf[el][j]`` is
    F_s(q_j, t) on the lag axis. Empty dicts where no edges or no q values
    were given.
    """

    lag_t_ps: np.ndarray
    lags: np.ndarray
    msd: dict[str, Series]
    msd4: dict[str, Series]
    alpha2: dict[str, Series]
    van_hove: dict[str, tuple[Histogram, ...]]
    gs_inv_ang3: dict[str, tuple[Series, ...]]
    isf: dict[str, tuple[Series, ...]]
    edges_r_ang: np.ndarray | None
    q_inv_ang: np.ndarray | None
    n_atoms: dict[str, int]
    remove_com_drift: bool
    block_length: int
    blocks: tuple[tuple[int, int], ...]
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        out = {"n_blocks": len(self.blocks),
               "frames per block": self.block_length,
               "lags (frames)": tuple(int(m) for m in self.lags),
               "centre-of-mass drift removed": self.remove_com_drift}
        if self.edges_r_ang is not None:
            out["van Hove edges (Å)"] = tuple(float(e) for e in
                                              (self.edges_r_ang[0],
                                               self.edges_r_ang[-1]))
            out["van Hove bins"] = int(self.edges_r_ang.size - 1)
        if self.q_inv_ang is not None:
            out["q (1/Å)"] = tuple(float(q) for q in self.q_inv_ang)
        return out

    def as_rows(self) -> list[dict]:
        containers: list = []
        for group in (self.msd, self.msd4, self.alpha2):
            containers += list(group.values())
        for group in (self.van_hove, self.gs_inv_ang3, self.isf):
            for items in group.values():
                containers += list(items)
        return _rows_of(containers)


def self_correlations(tracks: AtomTracks, lag_t_ps, *,
                      remove_com_drift: bool = False, edges_r_ang=None,
                      q_inv_ang=None, n_blocks: int = 1,
                      elements: Sequence[str] | None = None
                      ) -> SelfCorrelations:
    """alpha2(t), and optionally G_s(r, t) and F_s(q, t), at the lags listed.

    Every displacement over every origin of each block is used, at each lag in
    ``lag_t_ps`` (taken to the nearest whole frame, 1 frame or more). The van
    Hove histogram needs ``edges_r_ang``, the intermediate scattering
    function ``q_inv_ang``; neither has a default.
    """
    tracks = _check_tracks(tracks)
    if not isinstance(remove_com_drift, (bool, np.bool_)):
        raise ValueError("remove_com_drift is True or False")
    species = _species(tracks, elements)
    bounds, length, notes = _blocks(tracks.n_frames, n_blocks, tracks.frames,
                                    "self correlations")
    lags, lag_notes = _lags(lag_t_ps, tracks.dt_ps, length, 1)
    notes += lag_notes
    edges = None if edges_r_ang is None else _edges(edges_r_ang)
    binned = None if edges is None else _binner(edges)
    q = None if q_inv_ang is None else _q_values(q_inv_ang)
    n_b, n_l = len(bounds), lags.size
    rows = {el: tracks.rows_of(el) for el in species}
    count = {el: np.zeros((n_b, n_l)) for el in species}
    sum2 = {el: np.zeros((n_b, n_l)) for el in species}
    sum4 = {el: np.zeros((n_b, n_l)) for el in species}
    if edges is not None:
        hist = {el: np.zeros((n_b, n_l, edges.size - 1), np.int64)
                for el in species}
        below = {el: np.zeros((n_b, n_l), np.int64) for el in species}
        above = {el: np.zeros((n_b, n_l), np.int64) for el in species}
    if q is not None:
        sinc = {el: np.zeros((n_b, n_l, q.size)) for el in species}
    for b, (start, stop) in enumerate(bounds):
        positions = tracks.unwrapped_cart_ang[start:stop]
        com = tracks.com_ang[start:stop]
        chunk = max(1, CHUNK_VALUES // (3 * length))
        for el in species:
            for c0 in range(0, rows[el].size, chunk):
                part_ang = positions[:, rows[el][c0:c0 + chunk], :]
                if remove_com_drift:
                    part_ang = part_ang - com[:, None, :]
                for i, m in enumerate(lags):
                    step = part_ang[m:] - part_ang[:-m]
                    dr2_ang2 = np.einsum("tna,tna->tn", step, step).ravel()
                    count[el][b, i] += dr2_ang2.size
                    sum2[el][b, i] += dr2_ang2.sum()
                    sum4[el][b, i] += (dr2_ang2 * dr2_ang2).sum()
                    if edges is None and q is None:
                        continue
                    dr_ang = np.sqrt(dr2_ang2)
                    if edges is not None:
                        hist[el][b, i] += binned(dr_ang)
                        below[el][b, i] += int((dr_ang < edges[0]).sum())
                        above[el][b, i] += int((dr_ang > edges[-1]).sum())
                    if q is not None:
                        for j, q_value in enumerate(q):
                            sinc[el][b, i, j] += _sin_ratio_sum(dr_ang,
                                                                q_value)
    lag_t = lags * tracks.dt_ps
    labels = _block_labels(n_b)
    msd2, msd4, alpha2 = {}, {}, {}
    van_hove, gs, isf = {}, {}, {}
    for el in species:
        m2 = sum2[el] / count[el]
        m4 = sum4[el] / count[el]
        with np.errstate(invalid="ignore", divide="ignore"):
            a2 = np.where(m2 > 0, 3.0 * m4 / (5.0 * m2 * m2) - 1.0, np.nan)
        a2_notes = []
        if np.isnan(a2).any():
            a2_notes.append(f"alpha2 {el}: <dr^2> is 0 at some lag (no atom "
                            "moved), where alpha2 is undefined (NaN)")
        msd2[el] = Series(f"<dr^2> {el}", lag_t, "t_ps", "ps", "Å^2", m2,
                          frames=labels, row_kind="block")
        msd4[el] = Series(f"<dr^4> {el}", lag_t, "t_ps", "ps", "Å^4", m4,
                          frames=labels, row_kind="block")
        alpha2[el] = Series(
            f"alpha2 {el}", lag_t, "t_ps", "ps", "1", a2, frames=labels,
            row_kind="block",
            notes=["alpha2 = 3 <dr^4> / (5 <dr^2>^2) - 1 (Rahman 1964), from "
                   "the moments pooled over the atoms and origins of each "
                   "block"] + a2_notes)
        if edges is not None:
            shells = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
            centres = 0.5 * (edges[1:] + edges[:-1])
            hists, densities = [], []
            for i, t in enumerate(lag_t):
                hists.append(Histogram(
                    f"4 pi r^2 G_s(r, t = {t:.6g} ps) {el}", "Å", edges,
                    hist[el][:, i, :], below[el][:, i], above[el][:, i],
                    np.zeros(n_b, np.int64), density=True, frames=labels,
                    row_kind="block"))
                values = hist[el][:, i, :] / (count[el][:, i, None] * shells)
                densities.append(Series(
                    f"G_s(r, t = {t:.6g} ps) {el}", centres, "r_ang", "Å",
                    "1/Å^3", values, frames=labels, row_kind="block",
                    notes=["G_s per bin = count / (displacements x shell "
                           "volume), on the bin centres; every displacement, "
                           "in range or not, is in the normalisation"]))
            van_hove[el] = tuple(hists)
            gs[el] = tuple(densities)
        if q is not None:
            isf[el] = tuple(Series(
                f"F_s(q = {q_value:.6g} 1/Å, t) {el}", lag_t, "t_ps", "ps",
                "1", sinc[el][:, :, j] / count[el], frames=labels,
                row_kind="block",
                notes=["F_s = <sin(q |dr|) / (q |dr|)>, the average of "
                       "exp(i q . dr) over every direction of q"])
                for j, q_value in enumerate(q))
    notes.append("centre-of-mass drift " + ("removed" if remove_com_drift else
                                            "not removed"))
    return SelfCorrelations(
        lag_t_ps=_frozen(lag_t), lags=_frozen(lags), msd=msd2, msd4=msd4,
        alpha2=alpha2, van_hove=van_hove, gs_inv_ang3=gs, isf=isf,
        edges_r_ang=None if edges is None else _frozen(edges),
        q_inv_ang=None if q is None else _frozen(q),
        n_atoms={el: int(rows[el].size) for el in species},
        remove_com_drift=bool(remove_com_drift), block_length=length,
        blocks=_block_frames(tracks, bounds),
        notes=_unique_notes(list(tracks.notes) + notes))


# ---------------------------------------------------------------------------
# distinct van Hove function
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class DistinctVanHove:
    """g_d(r, t) = G_d / rho per element pair and lag; rows = time origins.

    ``g[(a, b)][i]`` is the Series at lag i on the bin centres
    (dimensionless; the partial g(r) at t = 0). ``n_origins[i]`` is how many
    origins lag i used.
    """

    lag_t_ps: np.ndarray
    lags: np.ndarray
    edges_r_ang: np.ndarray
    g: dict[tuple[str, str], tuple[Series, ...]]
    n_origins: tuple[int, ...]
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"lags (frames)": tuple(int(m) for m in self.lags),
                "time origins per lag": self.n_origins,
                "edges (Å)": (float(self.edges_r_ang[0]),
                              float(self.edges_r_ang[-1])),
                "bins": int(self.edges_r_ang.size - 1)}

    def as_rows(self) -> list[dict]:
        return _rows_of([s for items in self.g.values() for s in items])


def distinct_van_hove(tracks: AtomTracks, lag_t_ps, edges_r_ang,
                      pairs: Sequence[tuple[str, str]], *, n_origins: int
                      ) -> DistinctVanHove:
    """G_d(r, t) / rho_B for each (A, B) pair, ``n_origins`` origins per lag.

    The atoms of A at t0 and of B at t0 + t go into one frame in the box of
    t0, and :func:`.bulk.iter_pairs` (the one pair search FACET has) finds
    their pairs out to the last edge; the pairs of an atom with its own later
    position are left out. The box has to be the same at both times. Lags may
    be 0 (the partial g(r)). Origins are evenly spaced over the frames that
    reach the lag; fewer than ``n_origins`` when fewer frames do.
    """
    from . import bulk

    tracks = _check_tracks(tracks)
    origins_wanted = _whole(n_origins, "n_origins", 1)
    edges = _edges(edges_r_ang)
    lags, notes = _lags(lag_t_ps, tracks.dt_ps, tracks.n_frames, 0)
    if isinstance(pairs, (str, bytes)) or not pairs:
        raise ValueError("pairs needs one or more (A, B) element pairs")
    checked = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"pair {pair!r}: an (A, B) element pair is needed")
        a, b = (validate_symbol(str(pair[0])), validate_symbol(str(pair[1])))
        for el in (a, b):
            if el not in tracks.species:
                raise ValueError(f"{el} not in the tracks (they hold "
                                 f"{', '.join(tracks.species)})")
        if a == b and tracks.rows_of(a).size < 2:
            raise ValueError(f"{a}-{a} needs at least 2 atoms of {a}")
        checked.append((a, b))
    r_max_ang = float(edges[-1])
    if edges[0] > 0:
        notes.append(f"pairs closer than the first edge, "
                     f"{float(edges[0])!r} Å, are in no bin")
    shells = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    centres = 0.5 * (edges[1:] + edges[:-1])
    used = []
    out: dict[tuple[str, str], list[Series]] = {p: [] for p in checked}
    widths_min = None
    for m in lags:
        origins = np.unique(np.rint(np.linspace(
            0, tracks.n_frames - 1 - m, origins_wanted)).astype(np.int64))
        used.append(int(origins.size))
        for t0 in origins:
            if not np.array_equal(tracks.box_ang[t0], tracks.box_ang[t0 + m]):
                raise ValueError(
                    f"the box changes between trajectory frames "
                    f"{tracks.frames[t0]} and {tracks.frames[t0 + m]}; the "
                    "distinct van Hove function compares positions in one box")
        for a, b in checked:
            rows_a, rows_b = tracks.rows_of(a), tracks.rows_of(b)
            n_a, n_b = rows_a.size, rows_b.size
            partners = n_b - (1 if a == b else 0)
            values = np.empty((origins.size, edges.size - 1))
            coincident = 0
            for o, t0 in enumerate(origins):
                box = tracks.box_ang[t0]
                positions = tracks.unwrapped_cart_ang
                cart = np.concatenate([positions[t0, rows_a],
                                       positions[t0 + m, rows_b]])
                symbols = np.concatenate([np.full(n_a, a), np.full(n_b, b)])
                frame = frame_from_arrays(symbols, cart, box_ang=box)
                if widths_min is None:
                    widths_min = float(frame.perpendicular_widths_ang.min())
                counts = np.zeros(edges.size - 1, np.int64)
                for block in bulk.iter_pairs(frame, r_max_ang, d_min_ang=0.0,
                                             vectors_within_ang=0.0):
                    take = (block.i < n_a) & (block.j >= n_a)
                    if a == b:
                        take &= (block.j - n_a) != block.i
                    counts += np.histogram(block.d_ang[take], bins=edges)[0]
                    coincident += block.n_below_d_min
                rho = partners / abs(float(np.linalg.det(box)))
                values[o] = counts / (n_a * rho * shells)
            expected = 2 * n_a * origins.size if (a == b and m == 0) else 0
            series_notes = [
                f"g_d = pairs / (N_{a} x rho x shell volume), rho = (N_{b}"
                + (" - 1" if a == b else "") + ") / V: the number of distinct "
                f"{b} partners per unit volume; {origins.size} time origin(s), "
                "evenly spaced, each a row"]
            if coincident > expected:
                series_notes.append(
                    f"{coincident - expected} ordered pair(s) at 0 Å "
                    "(coincident positions) are in no bin")
            out[(a, b)].append(Series(
                f"g_d {a}-{b} (t = {m * tracks.dt_ps:.6g} ps)", centres,
                "r_ang", "Å", "1", values,
                frames=[int(v) for v in origins], row_kind="time origin",
                notes=series_notes))
    if widths_min is not None and r_max_ang > widths_min / 2:
        notes.append(f"the last edge, {r_max_ang!r} Å, lies beyond half the "
                     f"narrowest box width ({widths_min / 2:.6g} Å): pairs are "
                     "counted at every periodic image within it, as the "
                     "periodic model holds them")
    notes.append("distinct van Hove pairs from bulk.iter_pairs on one frame "
                 "holding A at t0 and B at t0 + t (subsampled origins: "
                 + ", ".join(f"{count} at {m * tracks.dt_ps:.6g} ps"
                             for count, m in zip(used, lags)) + ")")
    return DistinctVanHove(
        lag_t_ps=_frozen(lags * tracks.dt_ps), lags=_frozen(lags),
        edges_r_ang=_frozen(edges),
        g={p: tuple(v) for p, v in out.items()}, n_origins=tuple(used),
        notes=_unique_notes(list(tracks.notes) + notes))


# ---------------------------------------------------------------------------
# velocity autocorrelation and vibrational density of states
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class VACFResult:
    """VACF per element: ``c`` in Å^2/ps^2, ``c_norm`` divided by each
    block's own C(0); ``total_norm`` the mass-weighted sum over every atom of
    the tracks, normalised. Rows are blocks.

    ``c_norm_first_lag`` is the block-mean normalised VACF at t = dt, per
    element and ``'total'``: how far the velocities have decorrelated within
    one frame interval. Velocities sampled at intervals comparable to or
    longer than the vibrations' periods give values near 0 or below there,
    and their VDOS then holds power folded back from above the Nyquist
    frequency (module docstring)."""

    c: dict[str, Series]
    c_norm: dict[str, Series]
    total_norm: Series
    dt_ps: float
    velocity_source: str
    nyquist_thz: float
    n_atoms: dict[str, int]
    block_length: int
    blocks: tuple[tuple[int, int], ...]
    c_norm_first_lag: dict[str, float]
    notes: tuple[str, ...] = ()

    @property
    def lag_t_ps(self) -> np.ndarray:
        return self.total_norm.axis

    @property
    def method_parameters(self) -> dict:
        return {"velocities": self.velocity_source,
                "n_blocks": len(self.blocks),
                "frames per block": self.block_length,
                "lags": int(self.total_norm.axis.size)}

    def as_rows(self) -> list[dict]:
        return _rows_of(list(self.c.values()) + list(self.c_norm.values())
                        + [self.total_norm])


def _velocities(tracks: AtomTracks, source: str):
    if source == "file":
        if tracks.vel_ang_per_ps is None:
            raise ValueError(
                "the tracks hold no velocities (the file has none, or not in "
                "every frame); velocities='finite difference' derives them "
                "from the unwrapped positions, with the note that comes with "
                "it")
        return tracks.vel_ang_per_ps, tracks.frames, []
    if tracks.n_frames < 3:
        raise ValueError("central differences need at least 3 frames")
    unwrapped_ang = tracks.unwrapped_cart_ang
    vel_ang_per_ps = (unwrapped_ang[2:] - unwrapped_ang[:-2]) \
        / (2.0 * tracks.dt_ps)
    return vel_ang_per_ps, tracks.frames[1:-1], [
        "velocities by central differences of the unwrapped positions, "
        "(r(t + dt) - r(t - dt)) / (2 dt), asked for by the user: frames "
        f"{tracks.frames[0]} and {tracks.frames[-1]} have none, and a "
        "vibration of frequency nu is attenuated by sin(2 pi nu dt) / "
        "(2 pi nu dt) in amplitude (its square in the VACF and the VDOS); "
        f"dt = {tracks.dt_ps:.6g} ps"]


def vacf(tracks: AtomTracks, *, velocities: str = "file",
         max_lag_t_ps: float | None = None, n_blocks: int = 1,
         elements: Sequence[str] | None = None) -> VACFResult:
    """<v(t0) . v(t0 + t)> per element, every origin, by FFT."""
    tracks = _check_tracks(tracks)
    source = _choice(velocities, "velocities", VELOCITY_SOURCES)
    species = _species(tracks, elements)
    vel_ang_per_ps, labels_frames, notes = _velocities(tracks, source)
    bounds, length, block_notes = _blocks(vel_ang_per_ps.shape[0], n_blocks,
                                          labels_frames, "VACF")
    notes += block_notes
    n_lags, lag_notes = _lag_count(max_lag_t_ps, tracks.dt_ps, length)
    notes += lag_notes
    lag_t = np.arange(n_lags) * tracks.dt_ps
    # every element of the tracks is summed, so that the mass-weighted total
    # is sum_el m_el x (the element's sum): one element, one mass
    rows = {el: tracks.rows_of(el) for el in tracks.species}
    sums = {el: np.empty((len(bounds), n_lags)) for el in tracks.species}
    for b, (start, stop) in enumerate(bounds):
        part = vel_ang_per_ps[start:stop]
        for el in tracks.species:
            sums[el][b] = _acf_sums(part, n_lags, rows=rows[el])
    raw = {el: sums[el] / rows[el].size for el in species}
    total = sum(float(tracks.masses_amu[rows[el][0]]) * sums[el]
                for el in tracks.species)
    labels = _block_labels(len(bounds))
    c, c_norm = {}, {}
    for el in species:
        zero = raw[el][:, 0] == 0
        norm_notes = []
        if zero.any():
            norm_notes.append(f"VACF {el}: C(0) = 0 in {int(zero.sum())} "
                              "block(s) (every velocity 0); normalised value "
                              "undefined (NaN)")
        with np.errstate(invalid="ignore", divide="ignore"):
            normalised = np.where(zero[:, None], np.nan,
                                  raw[el] / raw[el][:, :1])
        c[el] = Series(f"VACF {el}", lag_t, "t_ps", "ps", "Å^2/ps^2",
                       raw[el], frames=labels, row_kind="block")
        c_norm[el] = Series(f"normalised VACF {el}", lag_t, "t_ps", "ps", "1",
                            normalised, frames=labels, row_kind="block",
                            notes=norm_notes)
    zero = total[:, 0] == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        total_norm = np.where(zero[:, None], np.nan, total / total[:, :1])
    total_series = Series(
        "normalised mass-weighted VACF, every atom of the tracks", lag_t,
        "t_ps", "ps", "1", total_norm, frames=labels, row_kind="block",
        notes=["sum_i m_i <v_i(0) . v_i(t)> / sum_i m_i <v_i(0) . v_i(0)>, "
               "masses the standard atomic weights"])
    nyquist = 1.0 / (2.0 * tracks.dt_ps)
    notes.append(f"Nyquist frequency 1/(2 dt) = {nyquist:.6g} THz: a "
                 "vibration above it is not removed by sampling every dt; it "
                 "folds back (aliases) to a frequency between 0 and "
                 f"{nyquist:.6g} THz, and the VDOS, normalised to 1 over that "
                 "band, holds its power there")
    first_lag = {el: float(c_norm[el].mean[1]) for el in species}
    first_lag["total"] = float(total_series.mean[1])
    notes.append(
        f"normalised VACF at the first lag (t = dt = {tracks.dt_ps:.6g} ps), "
        "block mean: " + ", ".join(
            f"{'mass-weighted total' if k == 'total' else k} {v:.4g}"
            for k, v in first_lag.items())
        + "; it is 1 at t = 0 by definition, and a value near 0 or below "
        "means the velocities decorrelate within one frame interval")
    return VACFResult(c=c, c_norm=c_norm, total_norm=total_series,
                      dt_ps=tracks.dt_ps, velocity_source=source,
                      nyquist_thz=nyquist,
                      n_atoms={el: int(rows[el].size) for el in species},
                      block_length=length,
                      blocks=tuple((labels_frames[s], labels_frames[e - 1])
                                   for s, e in bounds),
                      c_norm_first_lag=first_lag,
                      notes=_unique_notes(list(tracks.notes) + notes))


@dataclass(frozen=True, eq=False)
class VDOSResult:
    """g(nu) per element and for the mass-weighted total; rows are blocks.

    ``g_thz`` on ``freq_thz`` in 1/THz, ``g_inv_cm`` on ``wavenumber_inv_cm``
    in 1/cm^-1; each integrates to 1 from 0 to the Nyquist frequency. Keys
    are the elements and ``'total'``.
    """

    freq_thz: np.ndarray
    wavenumber_inv_cm: np.ndarray
    g_thz: dict[str, Series]
    g_inv_cm: dict[str, Series]
    window: str
    grid_spacing_thz: float
    nyquist_thz: float
    inv_cm_per_thz: float
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"VDOS taper": self.window,
                "frequency grid spacing (THz)": self.grid_spacing_thz}

    def as_rows(self) -> list[dict]:
        return _rows_of(list(self.g_thz.values())
                        + list(self.g_inv_cm.values()))


def vdos(vacf_result: VACFResult, *, window: str = "hann") -> VDOSResult:
    """Vibrational density of states from the normalised VACF (module docstring).

    ``window``: ``'hann'`` multiplies C by the half-Hann taper
    0.5 (1 + cos(pi m / (M - 1))), which reaches 0 at the last lag; ``'none'``
    leaves C as measured, and the cut at the last lag then shows as ripples.
    """
    if not isinstance(vacf_result, VACFResult):
        raise ValueError(f"a VACFResult is needed, not "
                         f"{type(vacf_result).__name__}")
    taper = _choice(window, "window", WINDOWS)
    sfft = _fft()
    constants = _constants()
    axis = vacf_result.lag_t_ps
    n_lags = axis.size
    if n_lags < 2:
        raise ValueError("the VDOS needs a VACF at 2 lags or more")
    dt_ps = vacf_result.dt_ps
    m = np.arange(n_lags)
    weights = 0.5 * (1.0 + np.cos(np.pi * m / (n_lags - 1))) \
        if taper == "hann" else np.ones(n_lags)
    freq_thz = m / ((2 * n_lags - 2) * dt_ps)
    inv_cm_per_thz = constants.tera / (constants.c / constants.centi)
    wavenumber_inv_cm = freq_thz * inv_cm_per_thz

    def transform(series: Series) -> np.ndarray:
        out = np.empty_like(series.per_frame)
        for k, row in enumerate(series.per_frame):
            tapered = row * weights
            even = np.concatenate([tapered, tapered[-2:0:-1]])
            out[k] = 2.0 * dt_ps * sfft.rfft(even).real[:n_lags]
        return out

    sources = dict(vacf_result.c_norm)
    sources["total"] = vacf_result.total_norm
    g_thz, g_cm = {}, {}
    for key, series in sources.items():
        values = transform(series)
        label = "mass-weighted total" if key == "total" else key
        common = [f"g = 4 x the trapezoid integral of the normalised VACF "
                  f"times cos(2 pi nu t), taper '{taper}'; integrates to 1 "
                  "from 0 to the Nyquist frequency"]
        g_thz[key] = Series(f"VDOS {label}", freq_thz, "freq_thz", "THz",
                            "1/THz", values,
                            frames=[int(v) for v in series.frames],
                            row_kind=series.row_kind, notes=common)
        g_cm[key] = Series(f"VDOS {label}", wavenumber_inv_cm,
                           "wavenumber_inv_cm", "cm^-1", "1/cm^-1",
                           values / inv_cm_per_thz,
                           frames=[int(v) for v in series.frames],
                           row_kind=series.row_kind, notes=common)
    notes = [f"frequency grid spacing 1/(2 (M - 1) dt) = {freq_thz[1]:.6g} "
             f"THz ({freq_thz[1] * inv_cm_per_thz:.6g} cm^-1) for M = "
             f"{n_lags} lags",
             f"1 THz = {inv_cm_per_thz:.10g} cm^-1 (scipy.constants: c = "
             f"{constants.c!r} m/s)"]
    return VDOSResult(freq_thz=_frozen(freq_thz),
                      wavenumber_inv_cm=_frozen(wavenumber_inv_cm),
                      g_thz=g_thz, g_inv_cm=g_cm, window=taper,
                      grid_spacing_thz=float(freq_thz[1]),
                      nyquist_thz=float(freq_thz[-1]),
                      inv_cm_per_thz=float(inv_cm_per_thz),
                      notes=_unique_notes(list(vacf_result.notes) + notes))


@dataclass(frozen=True, eq=False)
class GreenKuboDiffusion:
    """D = (1/3) x the integral of <v(t0) . v(t0 + t)> from 0 to t_max.

    ``running_ang2_per_ps`` is (1/3) x the trapezoid integral from 0 to every
    lag of the VACF (Å^2/ps, rows = blocks), so its plateau, or the lack of
    one, can be read. ``d_ang2_per_ps`` is its block mean at ``t_last_ps``,
    the last lag at or below the ``t_max_ps`` asked for; ``per_block`` the
    same value from each block. ``c_norm_at_end`` is the block-mean
    normalised VACF there: how far the VACF had decayed at the limit.
    ``blocks`` gives the first and last trajectory frame of each block (the
    VACF's; central-difference velocities start one frame later).
    """

    element: str
    running_ang2_per_ps: Series
    t_max_ps: float
    t_last_ps: float
    n_points: int
    d_ang2_per_ps: float
    d_m2_per_s: float
    c_norm_at_end: float
    per_block: Scalar
    velocity_source: str
    blocks: tuple[tuple[int, int], ...] = ()
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        return [{"descriptor": f"Green-Kubo diffusion coefficient "
                               f"{self.element}",
                 "D (Å^2/ps)": self.d_ang2_per_ps,
                 "D (m^2/s)": self.d_m2_per_s,
                 "std over blocks (Å^2/ps)": self.per_block.std,
                 "blocks": self.per_block.n_frames,
                 "upper limit (ps)": self.t_last_ps,
                 "normalised VACF at the limit": self.c_norm_at_end,
                 "velocities": self.velocity_source}] + \
            self.running_ang2_per_ps.as_rows()


def green_kubo_diffusion(vacf_result: VACFResult, t_max_ps: float, *,
                         elements: Sequence[str] | None = None
                         ) -> dict[str, GreenKuboDiffusion]:
    """D per element by the Green-Kubo route [18, 19], from :func:`vacf`.

    D = (1/3) x the trapezoid integral of the raw VACF (Å^2/ps^2) from 0 to
    ``t_max_ps`` (required: where the VACF has decayed and the integral
    levels off is a property of the system, read off the running integral,
    never assumed here). The tail beyond the limit is not in D; the
    normalised VACF at the limit says how far it had decayed. The running
    integral is given on every lag of the VACF.
    """
    if not isinstance(vacf_result, VACFResult):
        raise ValueError(f"a VACFResult is needed, not "
                         f"{type(vacf_result).__name__}")
    if isinstance(elements, str):
        raise ValueError("elements needs a collection of symbols, not one "
                         "string")
    axis = vacf_result.lag_t_ps
    index = _window(axis, 0.0, t_max_ps, vacf_result.dt_ps)
    end = int(index[-1])
    constants = _constants()
    to_si = constants.angstrom ** 2 / constants.pico
    wanted = list(vacf_result.c) if elements is None else \
        sorted({validate_symbol(str(e)) for e in elements})
    out = {}
    for el in wanted:
        if el not in vacf_result.c:
            raise ValueError(f"no VACF for {el}; the result holds "
                             f"{', '.join(vacf_result.c)}")
        series = vacf_result.c[el]
        rows = series.per_frame
        steps = 0.5 * (rows[:, 1:] + rows[:, :-1]) * np.diff(axis)[None, :]
        running = np.concatenate([np.zeros((rows.shape[0], 1)),
                                  np.cumsum(steps, axis=1)], axis=1) / 3.0
        labels = [int(v) for v in series.frames]
        running_series = Series(
            f"Green-Kubo running D {el}", axis, "t_ps", "ps", "Å^2/ps",
            running, frames=labels, row_kind=series.row_kind,
            notes=["(1/3) x the trapezoid integral of <v(t0) . v(t0 + t)> "
                   "from 0 to each lag"])
        values = running[:, end]
        d_ang2_per_ps = float(values.mean())
        c_end = float(vacf_result.c_norm[el].mean[end])
        notes = [
            f"D = (1/3) x the trapezoid integral of the VACF of {el} from 0 "
            f"to {axis[end]:.6g} ps ({end + 1} lags; the upper limit is the "
            "user's choice), the Green-Kubo route; the tail beyond is not in "
            f"D, and the normalised VACF there is {c_end:.4g}",
            "D is the mean over blocks of each block's integral; "
            "running_ang2_per_ps gives the integral at every lag"]
        if d_ang2_per_ps < 0:
            notes.append(f"the integral is {d_ang2_per_ps:.6g} Å^2/ps, below "
                         "0 at this limit: the VACF's negative lobe outweighs "
                         "its positive part there, and the value is not a "
                         "diffusion coefficient")
        per_block = Scalar(f"Green-Kubo D {el}", "Å^2/ps", values,
                           frames=labels, row_kind=series.row_kind)
        out[el] = GreenKuboDiffusion(
            element=el, running_ang2_per_ps=running_series,
            t_max_ps=float(t_max_ps), t_last_ps=float(axis[end]),
            n_points=end + 1, d_ang2_per_ps=d_ang2_per_ps,
            d_m2_per_s=d_ang2_per_ps * to_si,
            c_norm_at_end=c_end, per_block=per_block,
            velocity_source=vacf_result.velocity_source,
            blocks=tuple(vacf_result.blocks),
            notes=_unique_notes(notes + list(vacf_result.notes)))
    return out


def kinetic_temperature(tracks: AtomTracks) -> Scalar:
    """sum m v^2 / (3 n k_B) per frame, from the file's velocities, in K.

    Every atom of the tracks counts 3 degrees of freedom; a run that fixes the
    total momentum or holds constraints has fewer, and its thermostat
    temperature is then higher by 3n / (3n - n_constraints).
    """
    tracks = _check_tracks(tracks)
    if tracks.vel_ang_per_ps is None:
        raise ValueError("the tracks hold no velocities from the file")
    constants = _constants()
    v_si = constants.angstrom / constants.pico
    mass = tracks.masses_amu * constants.atomic_mass
    twice_ke = np.einsum("tna,tna,n->t", tracks.vel_ang_per_ps,
                         tracks.vel_ang_per_ps, mass) * v_si ** 2
    values = twice_ke / (3.0 * tracks.n_atoms * constants.k)
    return Scalar("kinetic temperature", "K", values,
                  frames=list(tracks.frames),
                  notes=list(_unique_notes(
                      [f"sum m v^2 / (3 n k_B) over the {tracks.n_atoms} "
                       "atoms of the tracks, 3 degrees of freedom each; "
                       + _masses_note(tracks.elements)]
                      + list(tracks.notes))))


# ---------------------------------------------------------------------------
# ionic conductivity
# ---------------------------------------------------------------------------

def _charges(charges_e, what: str) -> dict[str, float]:
    if not isinstance(charges_e, Mapping) or not charges_e:
        raise ValueError(f"{what}: charges_e needs a map element -> charge in "
                         "units of e (formal or force-field charges; the "
                         "choice is the user's, none is assumed)")
    return {validate_symbol(str(k)): _number(v, f"the charge of {k}")
            for k, v in charges_e.items()}


@dataclass(frozen=True, eq=False)
class NernstEinstein:
    """sigma = e^2 / (V k_B T) sum_a N_a z_a^2 D_a, in S/m, with every input.

    ``per_block`` is sigma from each block's D when every charged element's
    D came from a :class:`DiffusionFit` (or :class:`GreenKuboDiffusion`) over
    the same blocks of frames, which ``blocks`` gives (first and last
    trajectory frame of each; empty when there is no ``per_block``).
    """

    sigma_s_per_m: float
    contributions_s_per_m: dict[str, float]
    d_m2_per_s: dict[str, float]
    charges_e: dict[str, float]
    n_atoms: dict[str, int]
    volume_ang3: float
    temperature_k: float
    per_block: Scalar | None
    blocks: tuple[tuple[int, int], ...] = ()
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        rows = [{"descriptor": "Nernst-Einstein conductivity",
                 "sigma (S/m)": self.sigma_s_per_m,
                 "temperature (K)": self.temperature_k,
                 "volume (Å^3)": self.volume_ang3}]
        for el, value in self.contributions_s_per_m.items():
            rows.append({"descriptor": f"Nernst-Einstein contribution {el}",
                         "sigma (S/m)": value, "charge (e)": self.charges_e[el],
                         "atoms": self.n_atoms[el],
                         "D (m^2/s)": self.d_m2_per_s[el]})
        return rows


def nernst_einstein(fits: Mapping[str, DiffusionFit | GreenKuboDiffusion]
                    | None = None, *,
                    d_m2_per_s: Mapping[str, float] | None = None,
                    charges_e: Mapping[str, float], temperature_k: float,
                    n_atoms: Mapping[str, int], volume_ang3: float
                    ) -> NernstEinstein:
    """Nernst-Einstein conductivity of the elements given charges.

    The diffusion coefficients come as ``fits``, element ->
    :class:`DiffusionFit` or :class:`GreenKuboDiffusion` (each carries its
    own unit), and/or as ``d_m2_per_s``, element -> a bare number in m^2/s,
    the unit in the argument's name. A bare number in ``fits`` is refused: the
    module's fits report D in Å^2/ps as well, and a value in Å^2/ps read as
    m^2/s gives a conductivity 1e8 times too large with nothing to show it.
    ``charges_e``, ``temperature_k``, ``n_atoms`` (e.g.
    ``tracks.composition_model``) and ``volume_ang3`` (e.g.
    ``tracks.mean_volume_ang3``) are required. Every charged element needs a
    D and an atom count; a negative D is refused. Elements with a D and no
    charge are left out, with a note. The notes give every D used, in m^2/s,
    and where it came from.
    """
    charges = _charges(charges_e, "nernst_einstein")
    temperature = _positive(temperature_k, "temperature_k")
    volume = _positive(volume_ang3, "volume_ang3")
    if fits is None and d_m2_per_s is None:
        raise ValueError("no diffusion coefficient given: fits (element -> "
                         "DiffusionFit or GreenKuboDiffusion) or d_m2_per_s "
                         "(element -> D in m^2/s) states them")
    for name, value in (("fits", fits), ("d_m2_per_s", d_m2_per_s)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{name} needs a map element -> "
                             + ("DiffusionFit or GreenKuboDiffusion"
                                if name == "fits" else "D in m^2/s"))
    if not isinstance(n_atoms, Mapping):
        raise ValueError("n_atoms needs a map element -> atom count (e.g. "
                         "tracks.composition_model)")
    d_si: dict[str, float] = {}         # m^2/s, whichever route gave it
    origin: dict[str, str] = {}
    fitted: dict[str, DiffusionFit | GreenKuboDiffusion] = {}
    for key, value in (fits or {}).items():
        el = validate_symbol(str(key))
        if isinstance(value, DiffusionFit):
            origin[el] = (f"MSD fit over {value.t_first_ps:.6g} to "
                          f"{value.t_last_ps:.6g} ps")
        elif isinstance(value, GreenKuboDiffusion):
            origin[el] = (f"Green-Kubo integral to {value.t_last_ps:.6g} ps")
        else:
            raise ValueError(
                f"fits holds {value!r} for {el}, not a DiffusionFit or "
                "GreenKuboDiffusion; a bare number goes in d_m2_per_s, whose "
                "name states its unit (m^2/s; a fit's d_ang2_per_ps is in "
                "Å^2/ps, where the same D is a number 1e8 times larger)")
        fitted[el] = value
        d_si[el] = value.d_m2_per_s
    for key, value in (d_m2_per_s or {}).items():
        el = validate_symbol(str(key))
        if el in d_si:
            raise ValueError(f"{el} has a D in both fits and d_m2_per_s; one "
                             "is needed")
        d_si[el] = _number(value, f"D of {el}")
        origin[el] = "given in d_m2_per_s"
    counts = {validate_symbol(str(k)): _whole(v, f"atoms of {k}", 0)
              for k, v in n_atoms.items()}
    missing = [el for el in charges if el not in d_si]
    if missing:
        raise ValueError(f"no diffusion coefficient for {', '.join(missing)}, "
                         "which has a charge; every charged element needs one")
    missing = [el for el in charges if el not in counts]
    if missing:
        raise ValueError(f"no atom count for {', '.join(missing)}")
    negative = [el for el in charges if d_si[el] < 0]
    if negative:
        raise ValueError(
            "a diffusion coefficient below 0 for " + ", ".join(
                f"{el} ({d_si[el]:.6g} m^2/s, {origin[el]})"
                for el in negative)
            + ": the MSD falls across that fit window (or the VACF integral "
            "is negative at that limit), and the value is not a diffusion "
            "coefficient; a window where the MSD rises is needed")
    constants = _constants()
    prefactor = constants.e ** 2 / (volume * constants.angstrom ** 3
                                     * constants.k * temperature)
    contributions = {el: prefactor * counts[el] * charges[el] ** 2 * d_si[el]
                     for el in charges}
    sigma = math.fsum(contributions.values())
    notes = [
        "Nernst-Einstein: sigma = e^2 / (V k_B T) sum N z^2 D; every "
        "cross-correlation between ions is left out (a Haven ratio of 1 by "
        f"construction); e = {constants.e!r} C, k_B = {constants.k!r} J/K "
        "(scipy.constants)",
        f"T = {temperature!r} K and V = {volume!r} Å^3 as given",
        "D used (m^2/s): " + "; ".join(
            f"{el} {d_si[el]:.6g} ({origin[el]})" for el in sorted(charges))]
    left = sorted(set(d_si) - set(charges))
    if left:
        notes.append(f"{', '.join(left)}: a D was given but no charge, so "
                     "they are not in the sum")
    per_block = None
    blocks_used: tuple[tuple[int, int], ...] = ()
    if all(el in fitted for el in charges):
        sizes = {fitted[el].per_block.n_frames for el in charges}
        labels = {tuple(fitted[el].per_block.frames.tolist())
                  for el in charges}
        # per-block D of two elements are summed block by block only when
        # the blocks hold the same frames (block indices alone, as before,
        # paired blocks of different trajectories or frame ranges)
        spans = {tuple(fitted[el].blocks) for el in charges}
        if len(sizes) == 1 and len(labels) == 1 and sizes.pop() > 1:
            if len(spans) == 1:
                to_si = constants.angstrom ** 2 / constants.pico
                rows = np.zeros(len(next(iter(labels))))
                for el in charges:
                    rows = rows + prefactor * counts[el] * charges[el] ** 2 \
                        * fitted[el].per_block.per_frame * to_si
                per_block = Scalar("Nernst-Einstein conductivity", "S/m",
                                   rows, frames=list(labels.pop()),
                                   row_kind="block")
                blocks_used = spans.pop()
            else:
                notes.append(
                    "no per-block conductivity: the D of the charged "
                    "elements come from blocks of different frames ("
                    + "; ".join(f"{el}: {_block_spans(fitted[el].blocks)}"
                                for el in sorted(charges)) + ")")
    # what each D rests on (time axis, units, fit window, reader notes)
    for el in sorted(fitted):
        if el in charges:
            notes += list(fitted[el].notes)
    return NernstEinstein(
        sigma_s_per_m=float(sigma), contributions_s_per_m=contributions,
        d_m2_per_s={el: d_si[el] for el in charges}, charges_e=charges,
        n_atoms={el: counts[el] for el in charges}, volume_ang3=volume,
        temperature_k=temperature, per_block=per_block, blocks=blocks_used,
        notes=_unique_notes(notes))


@dataclass(frozen=True, eq=False)
class CollectiveConductivity:
    """sigma from the charge displacement M(t) = sum q_i r_i(t), in S/m.

    ``msd_charge`` is <|M(t0 + t) - M(t0)|^2> in e^2 Å^2 (rows = blocks);
    the fit is a least-squares line over the user's window. ``blocks`` gives
    the first and last trajectory frame of each block.
    """

    msd_charge: Series
    sigma_s_per_m: float
    slope_e2ang2_per_ps: float
    intercept_e2ang2: float
    t_min_ps: float
    t_max_ps: float
    n_points: int
    rms_residual_e2ang2: float
    loglog_slope: float
    per_block: Scalar
    charges_e: dict[str, float]
    temperature_k: float
    volume_ang3: float
    net_charge_e: float
    blocks: tuple[tuple[int, int], ...] = ()
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        return [{"descriptor": "collective conductivity",
                 "sigma (S/m)": self.sigma_s_per_m,
                 "std over blocks (S/m)": self.per_block.std,
                 "blocks": self.per_block.n_frames,
                 "fit window (ps)": f"{self.t_min_ps!r} to {self.t_max_ps!r}",
                 "lags fitted": self.n_points,
                 "rms residual (e^2 Å^2)": self.rms_residual_e2ang2,
                 "log-log slope": self.loglog_slope}] + \
            self.msd_charge.as_rows()


def collective_conductivity(tracks: AtomTracks, *,
                            charges_e: Mapping[str, float],
                            temperature_k: float, t_min_ps: float,
                            t_max_ps: float, n_blocks: int = 1,
                            max_lag_t_ps: float | None = None
                            ) -> CollectiveConductivity:
    """sigma = e^2 / (6 V k_B T) d<|dM|^2>/dt over the window (Helfand 1960).

    Needs every atom of the model in the tracks and a charge for every
    element. V is the mean box volume of the tracks.
    """
    tracks = _check_tracks(tracks)
    charges = _charges(charges_e, "collective_conductivity")
    temperature = _positive(temperature_k, "temperature_k")
    if tracks.n_atoms != tracks.n_atoms_model:
        raise ValueError(
            f"the tracks hold {tracks.n_atoms} of the model's "
            f"{tracks.n_atoms_model} atoms; the charge displacement sums over "
            "every ion, so collect_tracks without elements= is needed")
    missing = [el for el in tracks.species if el not in charges]
    if missing:
        raise ValueError(f"no charge for {', '.join(missing)}; the charge "
                         "displacement needs one for every element")
    unused = sorted(el for el in charges if el not in tracks.species)
    charges = {el: charges[el] for el in tracks.species}
    q = np.array([charges[str(s)] for s in tracks.elements])
    bounds, length, notes = _blocks(tracks.n_frames, n_blocks, tracks.frames,
                                    "charge displacement")
    if unused:
        notes.append(f"charges were given for {', '.join(unused)}, which the "
                     "tracks do not hold; they are not used and not in "
                     "charges_e")
    n_lags, lag_notes = _lag_count(max_lag_t_ps, tracks.dt_ps, length)
    notes += lag_notes
    lag_t = np.arange(n_lags) * tracks.dt_ps
    displacement = np.einsum("tna,n->ta", tracks.unwrapped_cart_ang, q)
    rows = np.empty((len(bounds), n_lags))
    for b, (start, stop) in enumerate(bounds):
        rows[b] = _msd_sums(displacement[start:stop, None, :], n_lags)
    series = Series("charge-displacement MSD", lag_t, "t_ps", "ps",
                    "e^2 Å^2", rows, frames=_block_labels(len(bounds)),
                    row_kind="block")
    fit = _window_fit(series, t_min_ps, t_max_ps, tracks.dt_ps)
    constants = _constants()
    volume = tracks.mean_volume_ang3
    factor = constants.e ** 2 * constants.angstrom ** 2 / constants.pico / (
        6.0 * volume * constants.angstrom ** 3 * constants.k * temperature)
    net = math.fsum(q.tolist())
    notes += [
        "collective conductivity: sigma = e^2 / (6 V k_B T) x the slope of "
        "the charge-displacement MSD, fitted by least squares to the "
        f"block-mean over {fit['t'][0]:.6g} to {fit['t'][-1]:.6g} ps (the "
        "window is the user's choice); it holds every cross-correlation "
        "between ions. M is one quantity for the whole model, not an "
        "average over atoms, so its MSD is averaged over time origins only",
        f"V = {volume!r} Å^3, the mean box volume over the frames; T = "
        f"{temperature!r} K as given; e = {constants.e!r} C, k_B = "
        f"{constants.k!r} J/K (scipy.constants)",
        f"log-log slope of the charge-displacement MSD over the window: "
        f"{fit['loglog']:.6g}"] + fit["notes"]
    scale = math.fsum(np.abs(q).tolist())
    if abs(net) > 1e-9 * scale:
        notes.append(f"the model's net charge with these charges is {net:.6g} "
                     "e, not 0, so M moves with the centre of the charge "
                     "distribution as well")
    if tracks.box_varies:
        notes.append("the box changes between frames; the mean volume is used")
    if fit["slope"] <= 0:
        notes.append(f"the fitted slope is {fit['slope']:.6g} e^2 Å^2/ps, not "
                     "above 0: no net charge transport is measured over this "
                     "window")
    per_block = Scalar("collective conductivity", "S/m",
                       fit["per_block_slope"] * factor, frames=fit["labels"],
                       row_kind="block")
    return CollectiveConductivity(
        msd_charge=series, sigma_s_per_m=float(fit["slope"] * factor),
        slope_e2ang2_per_ps=fit["slope"], intercept_e2ang2=fit["intercept"],
        t_min_ps=float(t_min_ps), t_max_ps=float(t_max_ps),
        n_points=int(fit["t"].size), rms_residual_e2ang2=fit["rms"],
        loglog_slope=fit["loglog"], per_block=per_block, charges_e=charges,
        temperature_k=temperature, volume_ang3=volume, net_charge_e=net,
        blocks=_block_frames(tracks, bounds),
        notes=_unique_notes(list(tracks.notes) + notes))


@dataclass(frozen=True, eq=False)
class HavenRatio:
    """H_R = sigma_NE / sigma_collective, with both inputs.

    ``haven_ratio`` is the ratio of the two block-mean conductivities.
    ``per_block`` holds each block's own ratio when both conductivities were
    measured over the same blocks of frames (None otherwise, with a note);
    its mean differs from ``haven_ratio`` (a mean of ratios is not a ratio
    of means), and its std is the spread of the per-block ratios.
    """

    haven_ratio: float
    sigma_ne_s_per_m: float
    sigma_collective_s_per_m: float
    per_block: Scalar | None
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        return [{"descriptor": "Haven ratio", "H_R": self.haven_ratio,
                 "sigma Nernst-Einstein (S/m)": self.sigma_ne_s_per_m,
                 "sigma collective (S/m)": self.sigma_collective_s_per_m,
                 "mean of per-block ratios": None if self.per_block is None
                 else self.per_block.mean,
                 "std of per-block ratios": None if self.per_block is None
                 else self.per_block.std}]


def haven_ratio(nernst: NernstEinstein,
                collective: CollectiveConductivity) -> HavenRatio:
    """The Haven ratio sigma_NE / sigma (Murch 1982); 1 when ions move
    independently, below 1 when their motions are positively correlated."""
    if not isinstance(nernst, NernstEinstein) or \
            not isinstance(collective, CollectiveConductivity):
        raise ValueError("a NernstEinstein and a CollectiveConductivity are "
                         "needed")
    notes = ["H_R = sigma_NE / sigma_collective (Murch 1982), the ratio of "
             "the two block-mean conductivities"]
    for label, a, b in (("temperature", nernst.temperature_k,
                         collective.temperature_k),
                        ("volume", nernst.volume_ang3, collective.volume_ang3)):
        if not math.isclose(a, b, rel_tol=1e-12):
            notes.append(f"the two conductivities use different {label}s "
                         f"({a!r} and {b!r})")
    ne_charges, co_charges = nernst.charges_e, collective.charges_e
    differ = sorted(el for el in ne_charges
                    if el in co_charges and co_charges[el] != ne_charges[el])
    if differ:
        notes.append("the charges differ for " + ", ".join(
            f"{el} ({ne_charges[el]:g} e in the Nernst-Einstein sum, "
            f"{co_charges[el]:g} e in the collective one)" for el in differ))
    ne_only = sorted(el for el in ne_charges if el not in co_charges)
    if ne_only:
        notes.append(f"{', '.join(ne_only)}: in the Nernst-Einstein sum, "
                     "absent from the collective one")
    co_only = sorted(el for el in co_charges
                     if el not in ne_charges and co_charges[el] != 0)
    if co_only:
        notes.append(
            f"{', '.join(co_only)}: charged in the collective sum, with no "
            "term in the Nernst-Einstein sum, so H_R sets the tracer terms of "
            f"{', '.join(sorted(ne_charges))} against the current of every "
            "charged element; their own motion is in sigma_collective only")
    sigma = collective.sigma_s_per_m
    if sigma > 0:
        ratio = nernst.sigma_s_per_m / sigma
    else:
        ratio = float("nan")
        notes.append(f"the collective conductivity is {sigma:.6g} S/m, not "
                     "above 0, so the ratio is undefined (NaN)")
    per_block = None
    same_blocks = tuple(nernst.blocks) == tuple(collective.blocks)
    if nernst.per_block is not None and not same_blocks:
        notes.append(
            "no per-block ratio: the Nernst-Einstein blocks ("
            + _block_spans(nernst.blocks) + ") and the collective ones ("
            + _block_spans(collective.blocks) + ") hold different frames")
    if nernst.per_block is not None and same_blocks and \
            nernst.per_block.n_frames == collective.per_block.n_frames and \
            np.array_equal(nernst.per_block.frames, collective.per_block.frames):
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(collective.per_block.per_frame > 0,
                              nernst.per_block.per_frame
                              / collective.per_block.per_frame, np.nan)
        per_block = Scalar("Haven ratio", "1", values,
                           frames=[int(v) for v in nernst.per_block.frames],
                           row_kind="block")
        notes.append(
            f"per_block holds each block's own ratio: their mean is "
            f"{per_block.mean:.6g} and H_R = {ratio:.6g} is the ratio of the "
            "block means (a mean of ratios is not a ratio of means); its std "
            "is the spread of the per-block ratios")
    return HavenRatio(haven_ratio=float(ratio),
                      sigma_ne_s_per_m=nernst.sigma_s_per_m,
                      sigma_collective_s_per_m=sigma, per_block=per_block,
                      notes=_unique_notes(notes))


# ---------------------------------------------------------------------------
# bond lifetimes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class BondTimeline:
    """Which cation-anion pairs are bonded at each frame.

    ``present`` (n_bonds, T) is True while the pair (``cation[b]``,
    ``anion[b]``), rows of the model, is bonded at ``v_bond_vu``. Every pair
    bonded in at least one frame has a row. ``elements`` are the model's.
    """

    source_path: str
    frames: tuple[int, ...]
    t_ps: np.ndarray
    dt_ps: float
    time_source: str
    elements: np.ndarray
    cation: np.ndarray
    anion: np.ndarray
    present: np.ndarray
    v_bond_vu: float
    notes: tuple[str, ...] = ()

    @property
    def n_frames(self) -> int:
        return int(self.t_ps.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.cation.shape[0])


def bond_timeline(bonds_per_frame: Sequence, *, elements, t_ps,
                  frames: Sequence[int] | None = None,
                  source_path: str = "<memory>", time_source: str = "t_ps given",
                  notes: Sequence[str] = ()) -> BondTimeline:
    """A timeline from :class:`.bulk.Bonds` already computed, one per frame.

    The route for a caller that holds the bonds of each frame (the glass
    descriptors compute them anyway), so no frame is analysed twice.
    ``elements`` are the model's symbols (rows of the frames), ``t_ps`` the
    frame times, evenly spaced. A pair bonded through two periodic images at
    once (possible only in a box under twice the bond length) is one bond.
    """
    from . import bulk

    if isinstance(notes, (str, bytes)):
        raise ValueError("notes needs a sequence of sentences, not one string")
    symbols = np.asarray(elements)
    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("elements needs one symbol per atom of the model")
    symbols = np.array([validate_symbol(str(s)) for s in symbols], dtype="<U2")
    times = np.asarray(t_ps, dtype=np.float64)
    bonds = list(bonds_per_frame)
    if times.shape != (len(bonds),):
        raise ValueError(f"{len(bonds)} frames of bonds and {times.size} "
                         "times; one time per frame is needed")
    labels = tuple(range(len(bonds))) if frames is None else \
        tuple(int(k) for k in frames)
    if len(labels) != len(bonds):
        raise ValueError("frames needs one label per frame of bonds")
    dt = _even_spacing(times, "bond timeline", labels)
    n = symbols.size
    keys, repeated = [], 0
    thresholds = set()
    cation_rows: set[int] = set()
    anion_rows: set[int] = set()
    for slot, item in enumerate(bonds):
        if not isinstance(item, bulk.Bonds):
            raise ValueError(f"frame {labels[slot]}: a bulk.Bonds is needed, "
                             f"not {type(item).__name__}")
        if len(item) and (int(item.cation.max()) >= n or
                          int(item.anion.max()) >= n):
            raise ValueError(f"frame {labels[slot]}: the bonds name rows "
                             f"beyond the {n} atoms given")
        if len(item) and (int(item.cation.min()) < 0 or
                          int(item.anion.min()) < 0):
            raise ValueError(f"frame {labels[slot]}: the bonds name a row "
                             "below 0; rows count the model's atoms from 0")
        thresholds.add(float(item.v_bond_vu))
        cation_rows.update(np.unique(item.cation).tolist())
        anion_rows.update(np.unique(item.anion).tolist())
        key = item.cation.astype(np.int64) * n + item.anion.astype(np.int64)
        unique = np.unique(key)
        repeated += key.size - unique.size
        keys.append(unique)
    both = sorted(cation_rows & anion_rows)
    if both:
        raise ValueError(
            f"row(s) {both[:5]} "
            f"({', '.join(str(symbols[row]) for row in both[:5])})"
            " are the cation of one bond and the anion of another; a bond "
            "joins a cation to an anion (bulk.bonds_at), and an atom is one "
            "or the other")
    if len(thresholds) > 1:
        raise ValueError(f"the frames' bonds were cut at different thresholds "
                         f"{sorted(thresholds)}; one bond definition is needed")
    every = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    present = np.zeros((every.size, len(bonds)), dtype=bool)
    for slot, unique in enumerate(keys):
        present[np.searchsorted(every, unique), slot] = True
    out_notes = list(notes)
    # with no oxidation states here, a pair whose two rows appear in no other
    # bond cannot be held to the cation/anion split above; what can be seen
    # is a bond between two atoms of one element, which one oxidation state
    # per element (md_model.model_oxidation) never gives
    alike = symbols[every // n] == symbols[every % n]
    if alike.any():
        kinds = sorted({str(s) for s in symbols[(every // n)[alike]]})
        out_notes.append(
            f"{int(alike.sum())} bonded pair(s) join two atoms of one element "
            f"({', '.join(f'{s}-{s}' for s in kinds)}): with one oxidation "
            "state per element a bond joins a cation to an anion of another "
            "element, so these come from per-atom states of opposite sign "
            "or from bonds built outside bulk.bonds_at; they are kept as "
            "given")
    if repeated:
        out_notes.append(f"{repeated} bond(s) joined a pair already bonded "
                         "through another periodic image in the same frame, "
                         "and count once")
    return BondTimeline(
        source_path=str(source_path), frames=labels, t_ps=_frozen(times),
        dt_ps=dt, time_source=time_source, elements=_frozen(symbols),
        cation=_frozen(every // n), anion=_frozen(every % n),
        present=_frozen(present),
        v_bond_vu=thresholds.pop() if thresholds else float("nan"),
        notes=_unique_notes(out_notes))


def collect_bonds(trajectory, ox, params: bv.ParameterSet | None = None, *,
                  frames: Sequence[int] | None = None,
                  timestep_fs: float | None = None,
                  frame_interval_ps: float | None = None,
                  v_bond_vu: float = bv.V_BOND_DEFAULT,
                  v_list_vu: float = bv.V_LIST_DEFAULT,
                  r_search_ang: float | None = None,
                  method: str = "auto") -> BondTimeline:
    """Bonds of every chosen frame: one :func:`.bulk.analyse_frame` each.

    ``ox`` is a :class:`.md_model.ModelOxidation` (or per-atom states). The
    bond is the CN's: v > ``v_bond_vu`` (:func:`.bulk.bonds_at`).
    """
    from . import bulk

    _tracking_check(trajectory)
    indices = _frame_list(trajectory, frames)
    if len(indices) < 2:
        raise ValueError("bond lifetimes need at least 2 frames")
    t_ps, source, notes = time_axis_ps(trajectory, indices,
                                       timestep_fs=timestep_fs,
                                       frame_interval_ps=frame_interval_ps)
    _even_spacing(t_ps, f"frames {indices[0]} to {indices[-1]}", indices)
    notes += [f"reader: {n}" for n in trajectory.notes]
    if trajectory.units_note and not any(trajectory.units_note in n
                                         for n in notes):
        notes.append(f"units as the reader gives them: "
                     f"{trajectory.units_note}")
    per_frame = []
    symbols = None
    first_notes: tuple[str, ...] | None = None
    differing: list[tuple[int, str]] = []
    for k in indices:
        frame = _load(trajectory, k)
        symbols = frame.elements
        ox_atom = ox.per_atom(frame.elements) if isinstance(
            ox, ModelOxidation) else ox
        table, _ = bulk.analyse_frame(frame, ox_atom, params,
                                      v_bond_vu=v_bond_vu, v_list_vu=v_list_vu,
                                      r_search_ang=r_search_ang, method=method)
        if first_notes is None:
            first_notes = table.notes
            notes.extend(f"frame {k}: {n}" for n in table.notes)
        else:
            extra = [n for n in table.notes if n not in first_notes]
            if extra:
                differing.append((k, extra[0]))
        per_frame.append(bulk.bonds_at(table, v_bond_vu))
    if differing:
        k, text = differing[0]
        notes.append(f"{len(differing)} later frame(s) carried bond-valence "
                     f"notes that frame {indices[0]} did not; the first, frame "
                     f"{k}: {text}")
    notes.append(f"bonds: cation-anion contacts with v > {v_bond_vu:g} v.u. "
                 "(bulk.bonds_at), one bulk.analyse_frame per frame")
    return bond_timeline(per_frame, elements=symbols, t_ps=t_ps,
                         frames=indices, source_path=trajectory.source_path,
                         time_source=source, notes=notes)


def _fill_gaps(present: np.ndarray, gap: int) -> np.ndarray:
    """Interior runs of False of length <= gap, between two True, set True."""
    n_b, length = present.shape
    padded = np.zeros((n_b, length + 2), dtype=np.int8)
    padded[:, 1:-1] = ~present
    change = np.diff(padded, axis=1)
    start_r, start_c = np.nonzero(change == 1)
    _, end_c = np.nonzero(change == -1)
    run = end_c - start_c
    interior = (start_c > 0) & (end_c < length) & (run <= gap)
    out = present.copy()
    if interior.any():
        rows = np.repeat(start_r[interior], run[interior])
        offsets = np.arange(rows.size) - np.repeat(
            np.cumsum(run[interior]) - run[interior], run[interior])
        out[rows, np.repeat(start_c[interior], run[interior]) + offsets] = True
    return out


def _continuous_sums(present: np.ndarray, n_lags: int) -> np.ndarray:
    """sum over bonds and origins of h(t0) H(t0, t0 + m), exactly."""
    n_b, length = present.shape
    padded = np.zeros((n_b, length + 2), dtype=np.int8)
    padded[:, 1:-1] = present
    change = np.diff(padded, axis=1)
    _, starts = np.nonzero(change == 1)
    _, ends = np.nonzero(change == -1)
    counts = np.bincount(ends - starts, minlength=length + 1).astype(np.int64)
    run = np.arange(length + 1, dtype=np.int64)
    # S0[j] = runs of length >= j, S1[j] = their total length
    s0 = np.cumsum(counts[::-1])[::-1]
    s1 = np.cumsum((run * counts)[::-1])[::-1]
    lags = np.arange(n_lags, dtype=np.int64)
    # a run of l frames holds max(0, l - m) origins at lag m
    return (s1[lags + 1] - lags * s0[lags + 1]).astype(np.int64)


def _intermittent_sums(present: np.ndarray, n_lags: int) -> np.ndarray:
    """sum over bonds and origins of h(t0) h(t0 + m), by FFT, rounded."""
    sfft = _fft()
    n_b, length = present.shape
    nfft = sfft.next_fast_len(2 * length - 1, real=True)
    total = np.zeros(n_lags)
    chunk = max(1, CHUNK_VALUES // nfft)
    for start in range(0, n_b, chunk):
        part = present[start:start + chunk].astype(np.float64)
        spectrum = sfft.rfft(part, n=nfft, axis=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).sum(axis=0)
        total += sfft.irfft(power, n=nfft)[:n_lags]
    rounded = np.rint(total)
    if np.abs(total - rounded).max() > 0.25:
        raise ValueError("the FFT sum of the bond products departed from a "
                         "whole number by more than 0.25; the timeline is "
                         "too long for this route")
    return rounded.astype(np.int64)


def _origin_counts(present: np.ndarray, n_lags: int) -> np.ndarray:
    """sum over bonds of h(t0) for t0 <= L - 1 - m."""
    length = present.shape[1]
    cumulative = np.cumsum(present.sum(axis=0, dtype=np.int64))
    return cumulative[length - 1 - np.arange(n_lags)]


@dataclass(frozen=True, eq=False)
class BondLifetimes:
    """C_C(t) and C_I(t) per (cation element, anion element); rows = blocks.

    ``bonds_per_frame[pair]`` is the number of such bonds in each frame;
    ``n_bonds_seen[pair]`` how many distinct pairs were bonded at least once.
    ``intermittent_uncorrelated[pair]`` (rows = blocks) is
    sum_i p_i^2 / sum_i p_i over the bonds seen in the block, p_i the fraction
    of the block's frames bond i is present: the value C_I takes when each
    bond's presence is uncorrelated in time at its measured fraction, which
    is where C_I levels off, above 0, while broken bonds form again.
    """

    lag_t_ps: np.ndarray
    continuous: dict[tuple[str, str], Series]
    intermittent: dict[tuple[str, str], Series]
    bonds_per_frame: dict[tuple[str, str], Scalar]
    n_bonds_seen: dict[tuple[str, str], int]
    gap_tolerance_frames: int
    v_bond_vu: float
    dt_ps: float
    block_length: int
    blocks: tuple[tuple[int, int], ...]
    intermittent_uncorrelated: dict[tuple[str, str], Scalar] = field(
        default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def method_parameters(self) -> dict:
        return {"bond threshold v_bond (v.u.)": self.v_bond_vu,
                "gap tolerance (frames)": self.gap_tolerance_frames,
                "n_blocks": len(self.blocks),
                "frames per block": self.block_length}

    def as_rows(self) -> list[dict]:
        return _rows_of(list(self.continuous.values())
                        + list(self.intermittent.values())
                        + list(self.intermittent_uncorrelated.values())
                        + list(self.bonds_per_frame.values()))


def bond_lifetimes(timeline: BondTimeline, *,
                   max_lag_t_ps: float | None = None, n_blocks: int = 1,
                   gap_tolerance_frames: int = 0) -> BondLifetimes:
    """Continuous and intermittent bond correlation functions (module docstring).

    ``gap_tolerance_frames`` (default 0, the strict continuous function)
    bridges breaks of up to that many frames between two frames where the
    bond holds, before C_C is formed; C_I never needs it. The bridged frames
    count as bonded wherever they fall, at origins and end points alike (the
    filled-history convention, module docstring).
    """
    if not isinstance(timeline, BondTimeline):
        raise ValueError(f"a BondTimeline is needed, not "
                         f"{type(timeline).__name__}")
    gap = _whole(gap_tolerance_frames, "gap_tolerance_frames", 0)
    bounds, length, notes = _blocks(timeline.n_frames, n_blocks,
                                    timeline.frames, "bond lifetimes")
    n_lags, lag_notes = _lag_count(max_lag_t_ps, timeline.dt_ps, length)
    notes += lag_notes
    lag_t = np.arange(n_lags) * timeline.dt_ps
    cation_el = timeline.elements[timeline.cation]
    anion_el = timeline.elements[timeline.anion]
    pairs = sorted({(str(c), str(a)) for c, a in zip(cation_el, anion_el)})
    if not pairs:
        raise ValueError("no bond in any frame of the timeline")
    labels = _block_labels(len(bounds))
    continuous, intermittent, per_frame, seen = {}, {}, {}, {}
    uncorrelated = {}
    for pair in pairs:
        select = (cation_el == pair[0]) & (anion_el == pair[1])
        h_all = timeline.present[select]
        seen[pair] = int(select.sum())
        per_frame[pair] = Scalar(
            f"{pair[0]}-{pair[1]} bonds per frame", "count",
            h_all.sum(axis=0).astype(np.float64), frames=list(timeline.frames))
        c_rows = np.empty((len(bounds), n_lags))
        i_rows = np.empty((len(bounds), n_lags))
        u_rows = np.empty(len(bounds))
        pair_notes = []
        for b, (start, stop) in enumerate(bounds):
            h = h_all[:, start:stop]
            h = h[h.any(axis=1)]
            if h.shape[0] == 0:
                c_rows[b] = np.nan
                i_rows[b] = np.nan
                u_rows[b] = np.nan
                pair_notes.append(f"{pair[0]}-{pair[1]}: no such bond in block "
                                  f"{b}; its row is NaN")
                continue
            held = h.sum(axis=1, dtype=np.int64)
            # sum p_i^2 / sum p_i with p_i = held_i / L, in whole numbers
            u_rows[b] = float((held * held).sum()) / (float(held.sum())
                                                       * h.shape[1])
            origins = _origin_counts(h, n_lags)
            with np.errstate(invalid="ignore", divide="ignore"):
                i_rows[b] = np.where(origins > 0, _intermittent_sums(
                    h, n_lags) / origins, np.nan)
                filled = _fill_gaps(h, gap) if gap else h
                filled_origins = _origin_counts(filled, n_lags)
                c_rows[b] = np.where(filled_origins > 0, _continuous_sums(
                    filled, n_lags) / filled_origins, np.nan)
        name = f"{pair[0]}-{pair[1]}"
        continuous[pair] = Series(
            f"C_C {name}", lag_t, "t_ps", "ps", "1", c_rows, frames=labels,
            row_kind="block", notes=pair_notes + [
                "C_C(t) = <h(t0) H(t0, t0 + t)> / <h(t0)>: the bond held at "
                "every frame from t0 to t0 + t"
                + (f", breaks of up to {gap} frame(s) bridged first: each "
                   "bridged frame counts as bonded, at an origin and at an "
                   "end point alike (the filled-history convention; counting "
                   "only origins and end points where the bond is present "
                   "gives other values)" if gap else "")
                + "; a break between two frames is not seen, so C_C depends "
                f"on the frame interval ({timeline.dt_ps:.6g} ps)"])
        intermittent[pair] = Series(
            f"C_I {name}", lag_t, "t_ps", "ps", "1", i_rows, frames=labels,
            row_kind="block", notes=pair_notes + [
                "C_I(t) = <h(t0) h(t0 + t)> / <h(t0)>: the bond present at "
                "t0 and at t0 + t, whatever happened between; it levels off "
                "near intermittent_uncorrelated, not at 0, while broken bonds "
                "form again"])
        uncorrelated[pair] = Scalar(
            f"C_I {name} for presence uncorrelated in time", "1", u_rows,
            frames=labels, row_kind="block",
            notes=pair_notes + [
                "sum_i p_i^2 / sum_i p_i over the bonds seen in each block, "
                "p_i the fraction of the block's frames bond i is present"])
    notes.append(f"a bond is a cation-anion pair with v > "
                 f"{timeline.v_bond_vu:g} v.u.; both functions are pooled over "
                 "the bonds of each element pair and every origin of a block")
    return BondLifetimes(
        lag_t_ps=_frozen(lag_t), continuous=continuous,
        intermittent=intermittent, bonds_per_frame=per_frame,
        n_bonds_seen=seen, gap_tolerance_frames=gap,
        v_bond_vu=timeline.v_bond_vu, dt_ps=timeline.dt_ps,
        block_length=length,
        blocks=tuple((timeline.frames[s], timeline.frames[e - 1])
                     for s, e in bounds),
        intermittent_uncorrelated=uncorrelated,
        notes=_unique_notes(list(timeline.notes) + notes))


@dataclass(frozen=True, eq=False)
class ResidenceTime:
    """Mean residence time from a bond correlation function, with its method.

    ``method`` 'integral': tau = trapezoid integral of the block-mean C from
    0 to ``t_max_ps``; ``amplitude`` None. 'exponential fit': C = A
    exp(-t / tau) fitted to ln C over the window. ``c_at_end`` is C at the
    last lag used. ``per_block`` holds tau from each block.
    ``c_uncorrelated`` is, for the intermittent function, the block mean of
    :attr:`BondLifetimes.intermittent_uncorrelated` (None for the continuous
    one, or when the lifetimes do not carry it).
    """

    pair: tuple[str, str]
    kind: str
    method: str
    tau_ps: float
    t_min_ps: float
    t_max_ps: float
    amplitude: float | None
    c_at_end: float
    rms_residual: float | None
    per_block: Scalar
    c_uncorrelated: float | None = None
    notes: tuple[str, ...] = ()

    def as_rows(self) -> list[dict]:
        return [{"descriptor": f"residence time {self.pair[0]}-{self.pair[1]}",
                 "function": self.kind, "method": self.method,
                 "tau (ps)": self.tau_ps, "std over blocks (ps)":
                 self.per_block.std, "blocks": self.per_block.n_frames,
                 "window (ps)": f"{self.t_min_ps!r} to {self.t_max_ps!r}",
                 "C at the end": self.c_at_end,
                 "C_I for uncorrelated presence": self.c_uncorrelated}]


def _trapezoid(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def residence_time(lifetimes: BondLifetimes, *, method: str,
                   kind: str = "continuous", t_min_ps: float | None = None,
                   t_max_ps: float | None = None,
                   pairs: Sequence[tuple[str, str]] | None = None
                   ) -> dict[tuple[str, str], ResidenceTime]:
    """tau per element pair, by the integral or an exponential fit (required).

    'integral' integrates from 0 to ``t_max_ps`` (default the last lag; a
    ``t_min_ps`` is refused); the tail beyond is not in it, and ``c_at_end``
    says how far C had fallen. 'exponential fit' needs both window ends and
    fits ln C (lags where C > 0) by least squares. On the intermittent
    function, which levels off above 0 while broken bonds form again, both
    carry a note with the measured level
    (:attr:`BondLifetimes.intermittent_uncorrelated`) and C_I at the end; the
    integral then grows with ``t_max_ps``.
    """
    if not isinstance(lifetimes, BondLifetimes):
        raise ValueError(f"a BondLifetimes is needed, not "
                         f"{type(lifetimes).__name__}")
    how = _choice(method, "method", RESIDENCE_METHODS)
    which = _choice(kind, "kind", BOND_KINDS)
    source = lifetimes.continuous if which == "continuous" else \
        lifetimes.intermittent
    if pairs is None:
        chosen = list(source)
    else:
        if isinstance(pairs, (str, bytes)):
            raise ValueError("pairs needs a sequence of (cation, anion) "
                             "element pairs")
        chosen = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)) or \
                    not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError(f"pair {pair!r}: a (cation, anion) element "
                                 "pair is needed, e.g. [('Si', 'O')]")
            chosen.append((validate_symbol(str(pair[0])),
                           validate_symbol(str(pair[1]))))
    axis = lifetimes.lag_t_ps
    out = {}
    for pair in chosen:
        if pair not in source:
            raise ValueError(f"no {pair[0]}-{pair[1]} bonds in the result")
        series = source[pair]
        notes = []
        level = None
        if which == "intermittent" and \
                pair in lifetimes.intermittent_uncorrelated:
            level = float(lifetimes.intermittent_uncorrelated[pair].mean)
        if how == "integral":
            if t_min_ps is not None:
                raise ValueError("the integral runs from t = 0; t_min_ps "
                                 "applies to the exponential fit")
            high = axis[-1] if t_max_ps is None else \
                _positive(t_max_ps, "t_max_ps")
            index = _window(axis, 0.0, high, lifetimes.dt_ps)
            t = axis[index]
            tau = _trapezoid(t, series.mean[index])
            blocks = np.array([_trapezoid(t, row[index])
                               for row in series.per_frame])
            amplitude, residual = None, None
            low = 0.0
            c_end = float(series.mean[index[-1]])
            if which == "continuous":
                notes.append(f"tau = trapezoid integral of the block-mean "
                             f"continuous function from 0 to {t[-1]:.6g} ps; "
                             f"C there is {c_end:.6g}, and the tail beyond is "
                             "not in tau")
            else:
                notes.append(
                    f"tau = trapezoid integral of the block-mean intermittent "
                    f"function from 0 to {t[-1]:.6g} ps, where C_I is "
                    f"{c_end:.6g}. C_I does not decay to 0 while broken bonds "
                    "form again: for presence uncorrelated in time it levels "
                    "off at sum p_i^2 / sum p_i"
                    + (f" = {level:.6g} here (measured, block mean)"
                       if level is not None else "")
                    + "; while C_I stays above 0 this integral grows with "
                    f"t_max_ps, by about {c_end:.4g} ps for every ps added to "
                    "the window, so it is a property of the window as much as "
                    "of the bonds")
        else:
            if t_min_ps is None or t_max_ps is None:
                raise ValueError("the exponential fit needs t_min_ps and "
                                 "t_max_ps: the window is the user's choice")
            index = _window(axis, t_min_ps, t_max_ps, lifetimes.dt_ps)
            low, high = float(t_min_ps), float(t_max_ps)

            def fit(values):
                keep = values[index] > 0
                if keep.sum() < 2:
                    return float("nan"), float("nan"), None
                slope, intercept, _ = _line_fit(axis[index][keep],
                                                np.log(values[index][keep]))
                if slope >= 0:
                    return float("nan"), math.exp(intercept), None
                tau_fit = -1.0 / slope
                model = math.exp(intercept) * np.exp(-axis[index] / tau_fit)
                return tau_fit, math.exp(intercept), float(
                    np.sqrt(np.mean((values[index] - model) ** 2)))

            tau, amplitude, residual = fit(series.mean)
            blocks = np.array([fit(row)[0] for row in series.per_frame])
            dropped = int((series.mean[index] <= 0).sum())
            c_end = float(series.mean[index[-1]])
            notes.append(f"tau from a least-squares line through ln C of the "
                         f"block-mean {which} function over {axis[index][0]:.6g}"
                         f" to {axis[index][-1]:.6g} ps (the window is the "
                         "user's choice)")
            if dropped:
                notes.append(f"{dropped} lag(s) in the window with C <= 0 have "
                             "no logarithm and are left out of the fit")
            if not math.isfinite(tau):
                notes.append("ln C does not fall across the window (or fewer "
                             "than 2 lags have C > 0): no time constant")
            if which == "intermittent":
                notes.append(
                    "C_I levels off above 0 while broken bonds form again"
                    + (f" (at sum p_i^2 / sum p_i = {level:.6g} for presence "
                       "uncorrelated in time, measured)" if level is not None
                       else "")
                    + f"; C_I at the end of the window is {c_end:.6g}, and "
                    "an exponential fitted to C_I there includes that level")
        per_block = Scalar(f"tau {pair[0]}-{pair[1]}", "ps", blocks,
                           frames=[int(v) for v in series.frames],
                           row_kind="block")
        # the function's own notes (a block with no such bond, NaN rows) and
        # the lifetimes' (bond threshold, time axis, reader) travel with tau
        out[pair] = ResidenceTime(
            pair=pair, kind=which, method=how, tau_ps=float(tau),
            t_min_ps=float(low), t_max_ps=float(high),
            amplitude=None if amplitude is None else float(amplitude),
            c_at_end=c_end, rms_residual=residual, per_block=per_block,
            c_uncorrelated=level,
            notes=_unique_notes(notes + list(series.notes)
                                + list(lifetimes.notes)))
    return out
