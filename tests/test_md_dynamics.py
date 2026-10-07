"""Dynamics from MD trajectories: every number checked against a known answer.

What is pinned, and why each check can fail:

* **The time axis is never assumed.** Times come from the file's timesteps
  times a user timestep, the file positions times a user interval, or the
  file's own times; with none of them, or frames that are not evenly spaced,
  the call is refused with the reason.
* **Unwrapping by continuity** recovers a random walk in a tilted box to
  1e-9 Å, follows the TOR equation of Bullerjahn et al. (2023, eq 2) written
  out independently in a breathing box, and refuses a step it cannot
  resolve, naming the atom.
* **MSD by FFT equals the direct double sum** over every origin to 1e-10 Å^2
  (lags, blocks); ballistic motion gives <v^2> t^2 and a log-log slope of 2;
  a random walk with known D gives 6 D t within five standard errors, the
  error measured from the independent atoms themselves.
* **Displacement statistics.** Gaussian displacements give alpha2 = 0 and
  F_s = exp(-q^2 D t) within five standard errors over independent groups of
  atoms; jumps of one fixed length a give alpha2 = -2/5 and F_s =
  sin(qa)/(qa) exactly; the van Hove histogram loses no displacement.
* **Distinct van Hove** at t = 0 on a simple cubic lattice counts 6, 12 and 8
  neighbours exactly, and equals a brute-force minimum-image pair count.
* **VACF and VDOS.** Circular motion gives a VACF of cos(w t) at every lag,
  the VDOS peak at w / (2 pi) and a VDOS integrating to 1; central-difference
  velocities carry the sin(w dt) / (w dt) attenuation stated in their note.
* **Conductivity** against a hand computation with the SI-exact e and k_B
  typed in here (not read from scipy); ions moving as one give a Haven ratio
  of 1/N exactly, neutral pairs moving together carry no current.
* **Bond lifetimes** equal a brute-force count on random bond histories, are
  1 for a bond that never breaks, follow p and p^m for bonds present at
  random, and a moving SiO4 unit read through bulk gives the expected
  history.
* **The file's own record is held to the axis.** With frame_interval_ps, a
  frame missing from the file, a restart appended to it or a 1 % change of
  dump interval in the timesteps is refused, and printed times are refused
  at half an interval and noted below it; an uneven axis is refused naming
  the frames on either side of the gap, and every refusal states the axis
  step it measured there; a record that is the same in every frame holds no
  time and is noted, not refused; the reader's notes and its unit sentence
  reach every result, Nernst-Einstein and residence times included.
* **Steps of a box width are refused** in positions handed over as
  continuous and in the file's unwrapped coordinates (image flags reset),
  naming the atom and the frames; continuity recovers the reset file.
* **What a number contains is stated.** fit_diffusion gives the centre of
  mass's own slope / 6 (equal to numpy.polyfit on com_msd) and its share of
  D; the VACF at the first lag is reported, and a 3.1 THz vibration sampled
  every 0.25 ps shows its folded peak at 0.9 THz; the integral of C_I comes
  with the level C_I keeps (sum p^2 / sum p, checked against a hand count
  and against C_I itself); Nernst-Einstein takes bare numbers only under
  d_m2_per_s; the Haven ratio names the elements on one side only and
  labels the per-block spread as that of ratios; per-block D, sigma_NE and
  H_R are paired only over blocks of the same frames, never by index alone.
* **Bond rows** below 0 or in both roles are refused, a bond between two
  atoms of one element is noted, and residence times carry the notes of
  the functions they were formed from.
* **Green-Kubo D** equals a hand trapezoid of (omega R)^2 cos(omega t) to
  1e-12 and the closed form for Ornstein-Uhlenbeck velocities within five
  standard errors over 20 independent groups.
* **No verdicts and no Qt.**

All inputs are synthetic; the real-model checks behind the regression tests
are recorded in the module docstring and were run outside the test suite.
"""
from __future__ import annotations

import ast
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from facet.core import bulk, md_dynamics as dyn, md_model
from facet.core.md_stats import Provenance, Series

ROOT = Path(__file__).resolve().parent.parent

# SI 2019 exact values, typed here so the hand computations do not lean on
# scipy.constants, which the module uses.
E_CHARGE_C = 1.602176634e-19
K_BOLTZMANN_J_PER_K = 1.380649e-23
SPEED_OF_LIGHT_M_PER_S = 299792458.0


def _trajectory(positions, elements, box, *, timesteps=None, times_ps=None,
                velocities=None, unwrapped=False, boxes=None, **reader):
    """A MemoryTrajectory of wrapped frames built from continuous positions.

    ``unwrapped`` True writes the positions as the file's unwrapped
    coordinates, or an array (T, n, 3) writes that array instead; ``reader``
    (notes, units_note, file_format) goes to the MemoryTrajectory."""
    frames = []
    for k in range(positions.shape[0]):
        if isinstance(unwrapped, np.ndarray):
            file_unwrapped = unwrapped[k]
        else:
            file_unwrapped = positions[k] if unwrapped else None
        frames.append(md_model.frame_from_arrays(
            elements, positions[k],
            box_ang=box if boxes is None else boxes[k],
            timestep=None if timesteps is None else int(timesteps[k]),
            time_ps=None if times_ps is None else float(times_ps[k]),
            unwrapped_cart_ang=file_unwrapped,
            vel_ang_per_ps=None if velocities is None else velocities[k]))
    return md_model.MemoryTrajectory(frames, **reader)


def _random_walk(rng, n_frames, n_atoms, d_ang2_per_ps, dt_ps, box_ang):
    steps = rng.normal(0.0, math.sqrt(2.0 * d_ang2_per_ps * dt_ps),
                       size=(n_frames, n_atoms, 3))
    steps[0] = rng.uniform(0.0, 1.0, size=(n_atoms, 3)) @ box_ang
    return np.cumsum(steps, axis=0)


def _direct_msd(x):
    """MSD(m) per atom by the definition: every origin, no FFT. (T, n)."""
    n_frames = x.shape[0]
    out = np.zeros((n_frames, x.shape[1]))
    for m in range(1, n_frames):
        step = x[m:] - x[:n_frames - m]
        out[m] = (step * step).sum(axis=2).mean(axis=0)
    return out


def _tracks(positions, elements, *, dt_ps=0.1, box=None, velocities=None):
    box = np.eye(3) * 50.0 if box is None else box
    t_ps = np.arange(positions.shape[0]) * dt_ps
    return dyn.tracks_from_arrays(elements, positions, t_ps=t_ps, box_ang=box,
                                  vel_ang_per_ps=velocities)


# ---------------------------------------------------------------------------
# the time axis
# ---------------------------------------------------------------------------

def _still(n_frames, *, timesteps=None, times_ps=None, **reader):
    positions = np.tile(np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]),
                        (n_frames, 1, 1))
    return _trajectory(positions, ["Na", "O"], np.eye(3) * 10.0,
                       timesteps=timesteps, times_ps=times_ps, **reader)


def test_timestep_fs_multiplies_the_file_timesteps():
    trajectory = _still(4, timesteps=[0, 100, 200, 300])
    t_ps, source, _ = dyn.time_axis_ps(trajectory, timestep_fs=2.0)
    assert np.allclose(t_ps, [0.0, 0.2, 0.4, 0.6], rtol=0, atol=1e-15)
    assert "timestep_fs = 2.0 fs" in source
    tracks = dyn.collect_tracks(trajectory, timestep_fs=2.0)
    assert tracks.dt_ps == pytest.approx(0.2, abs=1e-15)


def test_frame_interval_and_file_times_are_the_other_two_sources():
    trajectory = _still(3, times_ps=[5.0, 5.5, 6.0])
    t_ps, source, _ = dyn.time_axis_ps(trajectory)
    assert list(t_ps) == [5.0, 5.5, 6.0] and "written in the file" in source
    t_ps, source, notes = dyn.time_axis_ps(trajectory, frame_interval_ps=0.25)
    assert list(t_ps) == [0.0, 0.25, 0.5] and "frame_interval_ps" in source
    # the user's value is used, and the disagreement with the file is stated
    assert any("differ" in n and "2 times" in n for n in notes)


def test_a_file_written_in_fs_shows_the_factor_of_1000():
    trajectory = _still(3, timesteps=[0, 10, 20], times_ps=[0.0, 10.0, 20.0])
    _, _, notes = dyn.time_axis_ps(trajectory, timestep_fs=1.0)
    assert any("1000 times" in n for n in notes)


def test_no_time_axis_is_refused_with_both_ways_to_give_one():
    trajectory = _still(3)
    with pytest.raises(ValueError, match="no time axis") as error:
        dyn.collect_tracks(trajectory)
    assert "timestep_fs" in str(error.value)
    assert "frame_interval_ps" in str(error.value)
    with pytest.raises(ValueError, match="not both"):
        dyn.time_axis_ps(trajectory, timestep_fs=1.0, frame_interval_ps=1.0)
    with pytest.raises(ValueError, match="have none"):
        dyn.time_axis_ps(trajectory, timestep_fs=1.0)


def test_frames_not_evenly_spaced_or_not_increasing_are_refused():
    with pytest.raises(ValueError, match="not evenly spaced"):
        dyn.collect_tracks(_still(4, timesteps=[0, 100, 200, 350]),
                           timestep_fs=1.0)
    with pytest.raises(ValueError, match="do not increase"):
        dyn.collect_tracks(_still(3, times_ps=[0.0, 1.0, 1.0]))
    # printed times rounded to about six figures still form one grid
    dyn.collect_tracks(_still(3, times_ps=[0.0, 0.1000001, 0.2]))


def test_an_uneven_axis_is_refused_naming_the_frames_around_the_gap():
    # regression: the refusal used to name "position 1" whatever the gap
    with pytest.raises(ValueError, match="from frame 3 at 0.003 ps to frame 4 "
                                         "at 0.005 ps"):
        dyn.collect_tracks(_still(7, timesteps=[0, 1, 2, 3, 5, 6, 7]),
                           timestep_fs=1.0)
    t_ps = np.delete(np.arange(61) * 0.1, 30)      # frame 30 never written
    positions = np.zeros((60, 2, 3)) + 1.0
    with pytest.raises(ValueError, match="from frame 29 .* to frame 30 "):
        dyn.tracks_from_arrays(["Na", "O"], positions, t_ps=t_ps,
                               box_ang=np.eye(3) * 10.0)


def test_frame_interval_is_held_to_the_files_own_timesteps():
    # regression: file position x frame_interval_ps ignored the timesteps, so
    # a frame missing from the file, or a restart appended to it, was taken
    # as one more interval (an NS3 dump: D_Na 0.63 x the clean value)
    ok = dyn.time_axis_ps(_still(4, timesteps=[0, 50, 100, 150]),
                          frame_interval_ps=0.2)
    assert list(ok[0]) == pytest.approx([0.0, 0.2, 0.4, 0.6]) and ok[2] == []
    missing = _still(6, timesteps=[0, 100, 200, 400, 500, 600])
    with pytest.raises(ValueError, match="timesteps disagree") as error:
        dyn.collect_tracks(missing, frame_interval_ps=0.1)
    assert "frame 2 (timestep 200) to frame 3 (timestep 400)" in \
        str(error.value)
    # the same file with timestep_fs is refused by the even-grid check
    with pytest.raises(ValueError, match="not evenly spaced"):
        dyn.collect_tracks(missing, timestep_fs=1.0)
    # frames chosen on one side of the gap are accepted
    assert dyn.collect_tracks(missing, frames=[3, 4, 5],
                              frame_interval_ps=0.1).n_frames == 3
    restart = _still(6, timesteps=[0, 100, 200, 100, 200, 300])
    with pytest.raises(ValueError, match="timesteps do not increase: frame 3 "
                                         "has timestep 100 after frame 2"):
        dyn.collect_tracks(restart, frame_interval_ps=0.1)
    ox = md_model.model_oxidation(["Na", "O"])
    with pytest.raises(ValueError, match="timesteps disagree"):
        dyn.collect_bonds(missing, ox, frame_interval_ps=0.1)
    # a dump interval changed by 1 % is refused too (integer timesteps are
    # exact, so TIME_SPACING_RTOL applies)
    with pytest.raises(ValueError, match="timesteps disagree"):
        dyn.collect_tracks(_still(4, timesteps=[0, 100, 200, 301]),
                           frame_interval_ps=0.1)


def test_frame_interval_is_held_to_the_files_printed_times():
    # regression: with ITEM: TIME in the file, only the span was compared
    gap = _still(6, times_ps=[0.0, 0.1, 0.2, 0.4, 0.5, 0.6])
    with pytest.raises(ValueError, match="frame times disagree") as error:
        dyn.collect_tracks(gap, frame_interval_ps=0.1)
    assert "frame 2 (time 0.2 ps) to frame 3 (time 0.4 ps)" in str(error.value)
    # printed times are rounded: a departure under half an interval is a
    # note naming the frame, not a refusal
    rounded = _still(5, times_ps=[0.0, 0.1, 0.2003, 0.3, 0.4])
    tracks = dyn.collect_tracks(rounded, frame_interval_ps=0.1)
    assert any("frame times depart" in n and "frame 2" in n and "0.003" in n
               for n in tracks.notes)


def test_the_record_check_states_the_axis_step_it_measured():
    # regression: the refusal said "the axis steps by one interval there"
    # (and "places it one interval later") whatever the axis was; on the
    # file's own printed times the axis step is whatever the file printed
    shifted = _still(6, timesteps=[0, 200, 400, 600, 800, 1000],
                     times_ps=[0.0, 0.2, 0.4, 0.72, 0.8, 1.0])
    with pytest.raises(ValueError, match="timesteps disagree") as error:
        dyn.collect_tracks(shifted)
    text = str(error.value)
    assert "from frame 2 (timestep 400) to frame 3 (timestep 600)" in text
    assert "the axis steps by 0.32 ps where its median step is 0.2 ps" in text
    assert "frame 3 lies 0.6 of a frame interval" in text
    assert "one interval there" not in text
    restart = _still(4, timesteps=[0, 100, 50, 150],
                     times_ps=[0.0, 0.1, 0.3, 0.4])
    with pytest.raises(ValueError, match="timesteps do not increase") as error:
        dyn.collect_tracks(restart)
    assert "places it 0.2 ps later" in str(error.value)
    # with frame_interval_ps the axis step there is the interval, measured
    missing = _still(6, timesteps=[0, 100, 200, 400, 500, 600])
    with pytest.raises(ValueError, match="the axis steps by 0.1 ps"):
        dyn.collect_tracks(missing, frame_interval_ps=0.1)
    # regression: a file whose timesteps are all 0 holds no time, and was
    # refused as "not increasing" even with frame_interval_ps, the one way
    # left to state its axis; it is noted now, and timestep_fs still refused
    constant = _still(4, timesteps=[0, 0, 0, 0])
    tracks = dyn.collect_tracks(constant, frame_interval_ps=0.1)
    assert tracks.dt_ps == pytest.approx(0.1)
    assert any("every chosen frame carries the same timestep in the file "
               "(0)" in n for n in tracks.notes)
    with pytest.raises(ValueError, match="do not increase"):
        dyn.collect_tracks(constant, timestep_fs=1.0)


def test_the_xdatcar_configuration_number_assumption_is_noted():
    trajectory = _still(3, timesteps=[1, 2, 3], file_format="vasp-xdatcar")
    _, _, notes = dyn.time_axis_ps(trajectory, timestep_fs=2.0)
    assert any("configuration=" in n and "reference to verify" in n
               for n in notes)
    _, _, notes = dyn.time_axis_ps(trajectory, frame_interval_ps=0.002)
    assert not any("configuration=" in n for n in notes)


def test_the_readers_notes_and_units_reach_every_result():
    # regression: the reader's units caveat reached the tracks only on the
    # file-times route, and kinetic_temperature dropped every upstream note
    rng = np.random.default_rng(15)
    positions = _random_walk(rng, 12, 6, 0.2, 0.1, np.eye(3) * 12.0)
    velocities = rng.normal(size=positions.shape)
    units = "UNITS: velocities as written, Å/ps under units metal"
    trajectory = _trajectory(
        positions, ["Na"] * 3 + ["Cl"] * 3, np.eye(3) * 12.0,
        timesteps=np.arange(12), velocities=velocities, unwrapped=True,
        notes=["READER: timestep 5 repeated"], units_note=units)
    tracks = dyn.collect_tracks(trajectory, timestep_fs=100.0)
    msd = dyn.msd(tracks)
    vac = dyn.vacf(tracks)
    fits = dyn.fit_diffusion(msd, 0.2, 1.0)
    results = {
        "tracks": tracks.notes,
        "msd": msd.notes,
        "fit": fits["Na"].notes,
        # regression: the Nernst-Einstein sigma, whose D rest on the units
        # caveat, dropped every note of the fits it was given
        "nernst": dyn.nernst_einstein(
            fits, charges_e={"Na": 1, "Cl": -1}, temperature_k=500.0,
            n_atoms=tracks.composition_model,
            volume_ang3=tracks.mean_volume_ang3).notes,
        "self": dyn.self_correlations(tracks, [0.2]).notes,
        "vacf": vac.notes,
        "vdos": dyn.vdos(vac).notes,
        "green-kubo": dyn.green_kubo_diffusion(vac, 0.5)["Na"].notes,
        "temperature": dyn.kinetic_temperature(tracks).notes,
        "collective": dyn.collective_conductivity(
            tracks, charges_e={"Na": 1, "Cl": -1}, temperature_k=500.0,
            t_min_ps=0.2, t_max_ps=1.0).notes,
        "distinct": dyn.distinct_van_hove(tracks, [0.1], [1.0, 2.0],
                                          [("Na", "Cl")], n_origins=1).notes}
    for name, notes in results.items():
        assert any(units in n for n in notes), name
        assert any("reader: READER: timestep 5 repeated" in n
                   for n in notes), name
    # the masses note states what was read, not a difference never measured
    assert any("the file's own masses are not read" in n
               for n in results["temperature"])
    assert not any("force field's own" in n for n in results["temperature"])


# ---------------------------------------------------------------------------
# unwrapping
# ---------------------------------------------------------------------------

TILTED = np.array([[9.0, 0.0, 0.0], [3.1, 8.5, 0.0], [-2.2, 1.7, 10.4]])


def test_continuity_unwrap_recovers_a_random_walk_in_a_tilted_box():
    rng = np.random.default_rng(11)
    positions = _random_walk(rng, 120, 25, 0.2, 0.1, TILTED)
    trajectory = _trajectory(positions, ["Na"] * 10 + ["O"] * 15, TILTED,
                             timesteps=np.arange(120) * 50)
    tracks = dyn.collect_tracks(trajectory, timestep_fs=2.0)
    assert tracks.unwrap_method == "minimum image"
    moved = tracks.unwrapped_cart_ang - tracks.unwrapped_cart_ang[0]
    assert np.abs(moved - (positions - positions[0])).max() < 1e-9
    # the atoms did cross the box faces, so the unwrap was exercised
    frac = np.linalg.solve(TILTED.T, positions.reshape(-1, 3).T).T
    assert (np.floor(frac) != 0).any()
    steps = np.diff(positions, axis=0).reshape(-1, 3)
    expected = np.abs(np.linalg.solve(TILTED.T, steps.T)).max()
    assert tracks.max_step_fraction == pytest.approx(expected, rel=1e-9)


def test_the_files_unwrapped_coordinates_are_taken_as_given():
    rng = np.random.default_rng(12)
    positions = _random_walk(rng, 20, 6, 0.2, 0.1, np.eye(3) * 8.0)
    trajectory = _trajectory(positions, ["Li"] * 6, np.eye(3) * 8.0,
                             timesteps=np.arange(20), unwrapped=True)
    tracks = dyn.collect_tracks(trajectory, timestep_fs=100.0)
    assert tracks.unwrap_method == "file"
    assert np.array_equal(tracks.unwrapped_cart_ang, positions)
    with pytest.raises(ValueError, match="no unwrapped"):
        dyn.collect_tracks(_still(3, timesteps=[0, 1, 2]), timestep_fs=1.0,
                           unwrap="file")


def test_a_step_beyond_the_limit_is_refused_and_names_the_atom():
    positions = np.zeros((3, 2, 3)) + 1.0
    positions[1:, 1, 0] += 3.0                     # 0.3 of a 10 Å box at once
    trajectory = _trajectory(positions, ["Na", "Na"], np.eye(3) * 10.0,
                             timesteps=[0, 1, 2])
    with pytest.raises(ValueError, match="atom id 1") as error:
        dyn.collect_tracks(trajectory, timestep_fs=1.0)
    assert "step_limit_fraction" in str(error.value)
    tracks = dyn.collect_tracks(trajectory, timestep_fs=1.0,
                                step_limit_fraction=0.4)
    assert tracks.unwrapped_cart_ang[2, 1, 0] - \
        tracks.unwrapped_cart_ang[0, 1, 0] == pytest.approx(3.0, abs=1e-12)
    with pytest.raises(ValueError, match="above 0.5"):
        dyn.collect_tracks(trajectory, timestep_fs=1.0,
                           step_limit_fraction=0.6)


def test_wrapped_positions_handed_over_as_continuous_are_refused():
    # regression: tracks_from_arrays measured a 0.99-box step and accepted
    # it, and the MSD came out 2.5 times the true one with nothing in notes
    rng = np.random.default_rng(16)
    box = np.eye(3) * 10.0
    continuous = _random_walk(rng, 60, 16, 0.3, 0.1, box)
    wrapped = np.mod(continuous, 10.0)
    assert (wrapped != continuous).any()           # atoms crossed faces
    t_ps = np.arange(60) * 0.1
    with pytest.raises(ValueError, match=r"atom \d+ \(id \d+, Na\) moves "
                                         r"0\.\d+ of a box width between "
                                         r"frames \d+ and \d+, beyond "
                                         "step_limit_fraction = 0.25"):
        dyn.tracks_from_arrays(["Na"] * 16, wrapped, t_ps=t_ps, box_ang=box)
    tracks = dyn.tracks_from_arrays(["Na"] * 16, continuous, t_ps=t_ps,
                                    box_ang=box)
    steps = np.abs(np.diff(continuous, axis=0)).max() / 10.0
    assert tracks.max_step_fraction == pytest.approx(steps, rel=1e-12)
    assert any(f"{steps:.6g} of a box width (step_limit_fraction = 0.25"
               in n for n in tracks.notes)
    # a larger limit is an explicit statement, and above 0.5 is allowed here
    stated = dyn.tracks_from_arrays(["Na"] * 16, wrapped, t_ps=t_ps,
                                    box_ang=box, step_limit_fraction=1.0)
    assert any("step_limit_fraction = 1" in n for n in stated.notes)


def test_the_files_unwrapped_coordinates_are_held_to_the_step_limit():
    # regression: unwrap='file' measured the largest step and never compared
    # it with the limit; image flags reset at a restart gave D 6.4e4 x
    rng = np.random.default_rng(17)
    box = np.eye(3) * 10.0
    continuous = _random_walk(rng, 10, 6, 0.1, 0.1, box) + 20.0  # image 2
    reset = continuous.copy()
    reset[5:] = np.mod(continuous[5:], 10.0)       # image flags lost from 5
    trajectory = _trajectory(continuous, ["Na"] * 6, box,
                             timesteps=np.arange(10), unwrapped=reset)
    with pytest.raises(ValueError, match="between frames 4 and 5 in the "
                                         "file's unwrapped coordinates"):
        dyn.collect_tracks(trajectory, timestep_fs=100.0)
    # continuity does not use the image flags and recovers the motion
    tracks = dyn.collect_tracks(trajectory, timestep_fs=100.0,
                                unwrap="minimum image")
    moved = tracks.unwrapped_cart_ang - tracks.unwrapped_cart_ang[0]
    assert np.abs(moved - (continuous - continuous[0])).max() < 1e-9
    # for the file's coordinates a limit above 0.5 may be stated explicitly
    stated = dyn.collect_tracks(trajectory, timestep_fs=100.0, unwrap="file",
                                step_limit_fraction=3.0)
    assert stated.max_step_fraction > 1.0
    with pytest.raises(ValueError, match="above 0.5"):
        dyn.collect_tracks(trajectory, timestep_fs=100.0,
                           unwrap="minimum image", step_limit_fraction=0.6)


def _tor_1d(wrapped, lengths):
    """Bullerjahn et al. (2023) eq 2, as printed: u(i+1) = u(i) + dw -
    floor(dw / L(i+1) + 1/2) L(i+1), one axis of an orthorhombic box."""
    u = [wrapped[0]]
    for i in range(1, len(wrapped)):
        dw = wrapped[i] - wrapped[i - 1]
        u.append(u[-1] + dw - math.floor(dw / lengths[i] + 0.5) * lengths[i])
    return np.array(u)


def test_a_breathing_box_is_unwrapped_by_the_tor_scheme():
    rng = np.random.default_rng(13)
    n_frames, n_atoms = 80, 8
    lengths = 10.0 * (1.0 + 0.02 * np.sin(np.arange(n_frames) / 3.0))
    boxes = np.array([np.diag([L, L * 1.1, L * 0.9]) for L in lengths])
    positions = _random_walk(rng, n_frames, n_atoms, 0.3, 0.1, boxes[0])
    trajectory = _trajectory(positions, ["Na"] * n_atoms, None, boxes=boxes,
                             timesteps=np.arange(n_frames))
    tracks = dyn.collect_tracks(trajectory, timestep_fs=100.0)
    assert any("box changes" in n for n in tracks.notes)
    for atom in range(n_atoms):
        for axis in range(3):
            wrapped = [trajectory.frame(k).cart_ang[atom, axis]
                       for k in range(n_frames)]
            expected = _tor_1d(wrapped, boxes[:, axis, axis])
            assert np.abs(tracks.unwrapped_cart_ang[:, atom, axis]
                          - expected).max() < 1e-9
    noted = dyn.collect_tracks(
        _trajectory(positions, ["Na"] * n_atoms, None, boxes=boxes,
                    timesteps=np.arange(n_frames), unwrapped=True),
        timestep_fs=100.0)
    assert any("image counts" in n for n in noted.notes)


def test_a_file_whose_atoms_do_not_track_or_cannot_be_read_is_refused():
    trajectory = _still(3, timesteps=[0, 1, 2])
    trajectory.ids_track_atoms = False
    with pytest.raises(ValueError, match="ids_track_atoms"):
        dyn.collect_tracks(trajectory, timestep_fs=1.0)

    class Broken(md_model.MemoryTrajectory):
        def _load(self, k):
            if k == 2:
                raise md_model.FrameError("truncated")
            return super()._load(k)

    frames = [_still(4, timesteps=[0, 1, 2, 3]).frame(k) for k in range(4)]
    with pytest.raises(ValueError, match="frame 2 could not be read"):
        dyn.collect_tracks(Broken(frames), timestep_fs=1.0)


def test_the_centre_of_mass_is_over_every_atom_whatever_is_kept():
    rng = np.random.default_rng(14)
    positions = _random_walk(rng, 10, 9, 0.2, 0.1, np.eye(3) * 12.0)
    elements = ["Na"] * 3 + ["O"] * 6
    trajectory = _trajectory(positions, elements, np.eye(3) * 12.0,
                             timesteps=np.arange(10), unwrapped=True)
    every = dyn.collect_tracks(trajectory, timestep_fs=1.0)
    some = dyn.collect_tracks(trajectory, timestep_fs=1.0, elements=["Na"])
    assert some.n_atoms == 3 and some.n_atoms_model == 9
    assert np.array_equal(some.com_ang, every.com_ang)
    masses = np.array([22.98977] * 3 + [15.9994] * 6)     # gemmi 0.7.1 weights
    expected = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
    assert np.abs(every.com_ang - expected).max() < 1e-12
    assert not every.unwrapped_cart_ang.flags.writeable


# ---------------------------------------------------------------------------
# MSD and the diffusion coefficient
# ---------------------------------------------------------------------------

def test_fft_msd_equals_the_direct_double_sum():
    rng = np.random.default_rng(21)
    positions = _random_walk(rng, 300, 40, 0.5, 0.1, np.eye(3) * 30.0)
    elements = ["Na"] * 15 + ["O"] * 25
    tracks = _tracks(positions, elements)
    result = dyn.msd(tracks)
    direct = _direct_msd(positions)
    assert np.abs(result.msd["Na"].mean - direct[:, :15].mean(axis=1)).max() \
        < 1e-10
    assert np.abs(result.msd["O"].mean - direct[:, 15:].mean(axis=1)).max() \
        < 1e-10
    blocks = dyn.msd(tracks, n_blocks=3, max_lag_t_ps=5.0)
    assert blocks.msd["O"].per_frame.shape == (3, 51)
    for b in range(3):
        part = _direct_msd(positions[b * 100:(b + 1) * 100, 15:])
        assert np.abs(blocks.msd["O"].per_frame[b]
                      - part[:51].mean(axis=1)).max() < 1e-10


def test_ballistic_motion_gives_v2_t2_and_a_log_log_slope_of_2():
    rng = np.random.default_rng(22)
    n_frames, dt = 100, 0.1
    velocity = rng.normal(0.0, 0.4, size=(12, 3))
    start = rng.uniform(0.0, 15.0, size=(12, 3))
    positions = start[None] + np.arange(n_frames)[:, None, None] * dt \
        * velocity[None]
    trajectory = _trajectory(positions, ["Ar"] * 12, np.eye(3) * 15.0,
                             timesteps=np.arange(n_frames))
    tracks = dyn.collect_tracks(trajectory, timestep_fs=100.0)
    result = dyn.msd(tracks)
    t = result.lag_t_ps
    expected = (velocity ** 2).sum(axis=1).mean() * t ** 2
    assert np.abs(result.msd["Ar"].mean[1:] / expected[1:] - 1.0).max() < 1e-9
    fit = dyn.fit_diffusion(result, 1.0, 5.0)["Ar"]
    assert fit.loglog_slope == pytest.approx(2.0, abs=1e-9)


def test_a_random_walk_gives_6_d_t_within_five_standard_errors():
    rng = np.random.default_rng(23)
    d_true, dt = 0.1, 0.1
    positions = _random_walk(rng, 400, 400, d_true, dt, np.eye(3) * 40.0)
    tracks = _tracks(positions, ["Li"] * 400, dt_ps=dt)
    fit = dyn.fit_diffusion(dyn.msd(tracks), 0.5, 5.0)["Li"]
    # per-atom D from the direct MSD: the atoms are independent, so their
    # spread / sqrt(N) is the standard error of the mean D
    lags = np.arange(5, 51)
    per_atom = _direct_msd(positions[:, :, :])[lags]
    t = lags * dt
    slopes = ((t - t.mean())[:, None] * (per_atom - per_atom.mean(axis=0))
              ).sum(axis=0) / ((t - t.mean()) ** 2).sum()
    d_atoms = slopes / 6.0
    error = d_atoms.std(ddof=1) / math.sqrt(d_atoms.size)
    assert error < 0.05 * d_true                   # the check has power
    assert abs(fit.d_ang2_per_ps - d_true) < 5 * error
    # the fit of the mean MSD is the mean of the per-atom fits (OLS is linear)
    assert fit.d_ang2_per_ps == pytest.approx(d_atoms.mean(), rel=1e-9)
    assert fit.d_m2_per_s == pytest.approx(fit.d_ang2_per_ps * 1e-8, rel=1e-12)
    assert fit.n_points == 46 and fit.loglog_slope == pytest.approx(1, abs=0.1)
    assert math.isnan(fit.per_block.std)            # one block: NaN, noted
    assert any("one block" in n for n in fit.per_block.notes)


def test_centre_of_mass_drift_is_removed_when_asked_and_measured_always():
    rng = np.random.default_rng(24)
    start = rng.uniform(0.0, 20.0, size=(10, 3))
    drift = np.array([0.3, -0.2, 0.1])
    positions = start[None] + np.arange(50)[:, None, None] * 0.1 * drift
    tracks = _tracks(positions, ["Na"] * 4 + ["O"] * 6)
    plain = dyn.msd(tracks)
    removed = dyn.msd(tracks, remove_com_drift=True)
    t = plain.lag_t_ps
    assert np.abs(plain.msd["Na"].mean - (drift ** 2).sum() * t ** 2).max() \
        < 1e-10
    assert np.abs(plain.com_msd.mean - (drift ** 2).sum() * t ** 2).max() \
        < 1e-10
    assert np.abs(removed.msd["Na"].mean).max() < 1e-20
    assert removed.method_parameters["centre-of-mass drift removed"] is True


def test_fit_diffusion_reports_the_centre_of_mass_share_of_d():
    # regression: in a 300 K glass with net momentum 99 % of D_Si was the
    # drift of the whole model, and no DiffusionFit note said so
    rng = np.random.default_rng(26)
    positions = _random_walk(rng, 200, 30, 0.002, 0.1, np.eye(3) * 20.0)
    drift = np.cumsum(rng.normal(0.0, 0.2, size=(200, 3)), axis=0)
    positions = positions + drift[:, None, :]       # the whole model moves
    tracks = _tracks(positions, ["Si"] * 10 + ["O"] * 20)
    result = dyn.msd(tracks)
    fits = dyn.fit_diffusion(result, 2.0, 8.0)
    window = (result.lag_t_ps >= 2.0 - 1e-9) & (result.lag_t_ps <= 8.0 + 1e-9)
    by_hand = np.polyfit(result.lag_t_ps[window],
                         result.com_msd.mean[window], 1)[0] / 6.0
    for el, fit in fits.items():
        assert fit.com_d_ang2_per_ps == pytest.approx(by_hand, rel=1e-10)
        ratio = by_hand / fit.d_ang2_per_ps
        assert ratio > 0.5                           # the drift dominates here
        assert any("centre-of-mass drift not removed" in n
                   and f"{ratio:.4g} of this D" in n for n in fit.notes)
        assert fit.as_rows()[0]["centre of mass slope / 6 (Å^2/ps)"] == \
            fit.com_d_ang2_per_ps
        # the MSD's own notes (and through them the tracks') travel with D
        assert any("FFT algorithm" in n for n in fit.notes)
    last = result.com_msd.mean[-1]
    assert any(f"MSD is {last:.6g} Å^2" in n and "Si " in n
               for n in result.notes)
    removed = dyn.fit_diffusion(dyn.msd(tracks, remove_com_drift=True),
                                2.0, 8.0)
    assert all(f.com_d_ang2_per_ps is None for f in removed.values())
    assert all(any("removed before the MSD" in n for n in f.notes)
               for f in removed.values())
    assert removed["Si"].d_ang2_per_ps < 0.1 * fits["Si"].d_ang2_per_ps


def test_a_falling_msd_gives_a_negative_slope_that_is_noted_and_refused():
    n_frames, dt = 200, 0.05
    t = np.arange(n_frames) * dt
    positions = np.zeros((n_frames, 4, 3)) + 5.0
    positions[:, :, 0] += 0.3 * np.sin(2 * np.pi * t / 2.0)[:, None]
    tracks = _tracks(positions, ["Na"] * 4, dt_ps=dt)
    fit = dyn.fit_diffusion(dyn.msd(tracks), 1.2, 1.9)["Na"]
    assert fit.d_ang2_per_ps < 0
    assert any("below 0" in n for n in fit.notes)
    with pytest.raises(ValueError, match="below 0"):
        dyn.nernst_einstein({"Na": fit}, charges_e={"Na": 1.0},
                            temperature_k=300.0, n_atoms={"Na": 4},
                            volume_ang3=1000.0)


def test_fit_windows_are_checked():
    rng = np.random.default_rng(25)
    tracks = _tracks(_random_walk(rng, 30, 3, 0.1, 0.1, np.eye(3) * 9.0),
                     ["Na"] * 3)
    result = dyn.msd(tracks)
    with pytest.raises(ValueError, match="beyond the last lag"):
        dyn.fit_diffusion(result, 0.5, 5.0)
    with pytest.raises(ValueError, match="not above"):
        dyn.fit_diffusion(result, 1.0, 1.0)
    with pytest.raises(ValueError, match="at least 2"):
        dyn.fit_diffusion(result, 1.01, 1.05)
    with pytest.raises(ValueError, match="at least 2"):
        dyn.msd(tracks, n_blocks=20)


# ---------------------------------------------------------------------------
# alpha2, G_s, F_s
# ---------------------------------------------------------------------------

GROUPS = ("H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si",
          "P", "S", "Cl", "K", "Ca", "Ti", "Fe", "Zn")


def test_gaussian_displacements_give_alpha2_0_and_fs_exp_minus_q2_d_t():
    rng = np.random.default_rng(31)
    d_true, dt, per_group = 0.2, 0.1, 100
    positions = _random_walk(rng, 50, per_group * len(GROUPS), d_true, dt,
                             np.eye(3) * 40.0)
    elements = np.repeat(GROUPS, per_group)
    tracks = _tracks(positions, elements, dt_ps=dt)
    q = [0.8, 1.6]
    result = dyn.self_correlations(tracks, [0.1, 1.0, 4.0], q_inv_ang=q)
    # each group of atoms is independent of the others: the spread of the
    # group estimates over sqrt(groups) is the standard error of their mean
    alpha2 = np.array([result.alpha2[g].mean for g in GROUPS])
    mean, error = alpha2.mean(axis=0), alpha2.std(axis=0, ddof=1) / \
        math.sqrt(len(GROUPS))
    assert (error < 0.03).all()
    assert (np.abs(mean) < 5 * error).all()
    for j, q_value in enumerate(q):
        fs = np.array([result.isf[g][j].mean for g in GROUPS])
        mean, error = fs.mean(axis=0), fs.std(axis=0, ddof=1) / \
            math.sqrt(len(GROUPS))
        expected = np.exp(-q_value ** 2 * d_true * result.lag_t_ps)
        assert (np.abs(mean - expected) < 5 * error).all()


def test_jumps_of_one_length_give_alpha2_minus_two_fifths_exactly():
    rng = np.random.default_rng(32)
    a = 0.7
    direction = rng.normal(size=(500, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    start = rng.uniform(0.0, 30.0, size=(500, 3))
    positions = np.stack([start, start + a * direction])
    tracks = _tracks(positions, ["Na"] * 500)
    edges = np.array([0.0, 0.5, 0.65, 0.75, 1.0])
    result = dyn.self_correlations(tracks, [0.1], edges_r_ang=edges,
                                   q_inv_ang=[1.0, 3.0])
    assert result.alpha2["Na"].mean[0] == pytest.approx(-0.4, abs=1e-12)
    assert result.msd["Na"].mean[0] == pytest.approx(a * a, rel=1e-12)
    for j, q_value in enumerate((1.0, 3.0)):
        assert result.isf["Na"][j].mean[0] == pytest.approx(
            math.sin(q_value * a) / (q_value * a), abs=1e-12)
    histogram = result.van_hove["Na"][0]
    assert histogram.counts.tolist() == [[0, 0, 500, 0]]


def test_the_van_hove_histogram_loses_no_displacement():
    rng = np.random.default_rng(33)
    positions = _random_walk(rng, 40, 50, 0.5, 0.1, np.eye(3) * 30.0)
    tracks = _tracks(positions, ["O"] * 50)
    edges = np.linspace(0.2, 1.5, 14)
    result = dyn.self_correlations(tracks, [0.3, 1.0], edges_r_ang=edges,
                                   n_blocks=2)
    for i, m in enumerate(result.lags):
        histogram = result.van_hove["O"][i]
        for b in range(2):                         # numpy.histogram, explicit
            part = positions[b * 20:(b + 1) * 20]
            r = np.linalg.norm(part[m:] - part[:-m], axis=2).ravel()
            assert histogram.counts[b].tolist() == \
                np.histogram(r, bins=edges)[0].tolist()
        samples = 50 * (20 - m)                    # atoms x origins, per block
        assert (histogram.counts.sum(axis=1) + histogram.n_below
                + histogram.n_above == samples).all()
        assert histogram.n_below.sum() > 0 and histogram.n_above.sum() > 0
        inside = histogram.counts.sum(axis=1) / samples
        assert np.allclose((histogram.per_frame * np.diff(edges)).sum(axis=1),
                           inside, rtol=1e-12)
        shells = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
        assert np.allclose((result.gs_inv_ang3["O"][i].per_frame * shells)
                           .sum(axis=1), inside, rtol=1e-12)


def test_atoms_that_never_move_give_fs_1_and_an_undefined_alpha2():
    positions = np.tile(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                        (5, 1, 1))
    tracks = _tracks(positions, ["Si", "Si"])
    result = dyn.self_correlations(tracks, [0.1, 0.3], edges_r_ang=[0.0, 0.1],
                                   q_inv_ang=[2.0])
    assert result.isf["Si"][0].mean.tolist() == [1.0, 1.0]
    assert np.isnan(result.alpha2["Si"].mean).all()
    assert any("undefined" in n for n in result.alpha2["Si"].notes)
    assert result.van_hove["Si"][1].counts.tolist() == [[4]]


def test_lags_are_whole_frames_and_out_of_range_lags_are_refused():
    rng = np.random.default_rng(34)
    tracks = _tracks(_random_walk(rng, 20, 3, 0.1, 0.1, np.eye(3) * 9.0),
                     ["Na"] * 3)
    result = dyn.self_correlations(tracks, [0.26, 0.3])
    assert result.lags.tolist() == [3]
    assert any("nearest whole frame" in n for n in result.notes)
    assert any("merged" in n for n in result.notes)
    with pytest.raises(ValueError, match="smallest lag"):
        dyn.self_correlations(tracks, [0.0])
    with pytest.raises(ValueError, match="reaches"):
        dyn.self_correlations(tracks, [2.5])
    assert dyn.log_spaced_lag_times(tracks, 5).tolist() == pytest.approx(
        [0.1, 0.2, 0.4, 0.9, 1.9])


# ---------------------------------------------------------------------------
# distinct van Hove
# ---------------------------------------------------------------------------

def test_distinct_van_hove_at_t0_counts_the_simple_cubic_shells():
    a, side = 2.0, 6
    grid = np.indices((side, side, side)).reshape(3, -1).T * a + 0.5
    positions = np.stack([grid, grid])
    tracks = _tracks(positions, ["Ar"] * grid.shape[0],
                     box=np.eye(3) * a * side)
    edges = np.array([1.9, 2.1, 2.7, 2.9, 3.3, 3.6])
    result = dyn.distinct_van_hove(tracks, [0.0], edges, [("Ar", "Ar")],
                                   n_origins=2)
    g = result.g[("Ar", "Ar")][0]
    rho = (grid.shape[0] - 1) / (a * side) ** 3
    shells = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    neighbours = g.mean * rho * shells
    assert np.abs(neighbours - [6, 0, 12, 0, 8]).max() < 1e-9
    assert g.std.tolist() == [0.0] * 5            # two identical origins


def test_distinct_van_hove_equals_a_brute_force_pair_count():
    rng = np.random.default_rng(41)
    box = 10.0
    # continuous positions with large moves (up to 0.2 of the box per frame
    # and axis, under the step limit), so A at t0 and B at t0 + 2 frames are
    # far from their t0 arrangement; the brute force reads them mod the box
    positions = np.cumsum(np.concatenate([
        rng.uniform(0.0, box, size=(1, 50, 3)),
        rng.uniform(-0.2 * box, 0.2 * box, size=(6, 50, 3))]), axis=0)
    elements = ["Na"] * 20 + ["O"] * 30
    tracks = _tracks(positions, elements, box=np.eye(3) * box)
    edges = np.linspace(0.5, 4.5, 9)
    result = dyn.distinct_van_hove(tracks, [0.2], edges, [("Na", "O")],
                                   n_origins=3)
    origins = np.unique(np.rint(np.linspace(0, 6 - 2, 3)).astype(int))
    assert result.n_origins == (origins.size,)
    shells = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    for row, t0 in enumerate(origins):
        a = positions[t0, :20]
        b = positions[t0 + 2, 20:]
        delta = b[None, :, :] - a[:, None, :]
        delta -= box * np.round(delta / box)        # minimum image, r < L/2
        counts = np.histogram(np.linalg.norm(delta, axis=2).ravel(),
                              bins=edges)[0]
        expected = counts / (20 * (30 / box ** 3) * shells)
        assert np.abs(result.g[("Na", "O")][0].per_frame[row]
                      - expected).max() < 1e-12


def test_distinct_van_hove_refuses_a_box_that_changes():
    positions = np.zeros((3, 2, 3)) + 1.0
    positions[:, 1] += 2.0
    boxes = np.array([np.eye(3) * 10.0, np.eye(3) * 10.5, np.eye(3) * 10.0])
    tracks = dyn.tracks_from_arrays(["Na", "O"], positions,
                                    t_ps=[0.0, 0.1, 0.2], box_ang=boxes)
    with pytest.raises(ValueError, match="box changes"):
        dyn.distinct_van_hove(tracks, [0.1], [0.5, 3.0], [("Na", "O")],
                              n_origins=2)


# ---------------------------------------------------------------------------
# VACF, VDOS, kinetic temperature
# ---------------------------------------------------------------------------

def _circular(n_frames, dt, omega, radius=0.3, n_atoms=6):
    """Atoms on circles: v(t0) . v(t0 + t) = (omega R)^2 cos(omega t) for every
    origin, so the VACF is cos(omega t) exactly, not only on average."""
    rng = np.random.default_rng(51)
    t = np.arange(n_frames) * dt
    phase = rng.uniform(0, 2 * np.pi, size=n_atoms)
    angle = omega * t[:, None] + phase[None, :]
    centre = rng.uniform(2.0, 18.0, size=(n_atoms, 3))
    positions = np.repeat(centre[None], n_frames, axis=0)
    positions[:, :, 0] += radius * np.cos(angle)
    positions[:, :, 1] += radius * np.sin(angle)
    velocities = np.zeros_like(positions)
    velocities[:, :, 0] = -radius * omega * np.sin(angle)
    velocities[:, :, 1] = radius * omega * np.cos(angle)
    return positions, velocities


def test_circular_motion_gives_a_vacf_of_cos_omega_t():
    dt, n_lags = 0.01, 101
    nu = 10 / ((2 * n_lags - 2) * dt)              # on the VDOS grid
    omega = 2 * np.pi * nu
    positions, velocities = _circular(400, dt, omega)
    tracks = _tracks(positions, ["Si"] * 6, dt_ps=dt, velocities=velocities)
    result = dyn.vacf(tracks, max_lag_t_ps=(n_lags - 1) * dt)
    t = result.lag_t_ps
    assert np.abs(result.c_norm["Si"].mean - np.cos(omega * t)).max() < 1e-12
    assert result.c["Si"].mean[0] == pytest.approx((0.3 * omega) ** 2,
                                                   rel=1e-12)
    assert np.abs(result.total_norm.mean - np.cos(omega * t)).max() < 1e-12
    assert result.nyquist_thz == pytest.approx(50.0)
    for window in dyn.WINDOWS:
        spectrum = dyn.vdos(result, window=window)
        g = spectrum.g_thz["Si"].mean
        assert int(np.argmax(g)) == 10
        assert spectrum.freq_thz[10] == pytest.approx(nu, rel=1e-12)
        # the trapezoid integral over 0 .. Nyquist is C(0) = 1
        freq = spectrum.freq_thz
        integral = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(freq)))
        assert integral == pytest.approx(1.0, abs=1e-12)
        per_inv_cm = 1e12 / (SPEED_OF_LIGHT_M_PER_S * 100.0)
        assert spectrum.inv_cm_per_thz == pytest.approx(per_inv_cm, rel=1e-15)
        assert np.allclose(spectrum.wavenumber_inv_cm, freq * per_inv_cm,
                           rtol=1e-15, atol=0)
        assert np.allclose(spectrum.g_inv_cm["Si"].mean, g / per_inv_cm,
                           rtol=1e-15, atol=0)


def test_the_mass_weighted_vacf_weighs_each_element_by_its_mass():
    dt = 0.01
    omega_si, omega_o = 2 * np.pi * 4.0, 2 * np.pi * 9.0
    p_si, v_si = _circular(300, dt, omega_si, radius=0.2, n_atoms=3)
    p_o, v_o = _circular(300, dt, omega_o, radius=0.3, n_atoms=5)
    tracks = _tracks(np.concatenate([p_si, p_o], axis=1), ["Si"] * 3 + ["O"] * 5,
                     dt_ps=dt, velocities=np.concatenate([v_si, v_o], axis=1))
    result = dyn.vacf(tracks, max_lag_t_ps=1.0)
    t = result.lag_t_ps
    m_si, m_o = 28.0855, 15.9994                   # gemmi 0.7.1 weights
    weight_si = m_si * 3 * (0.2 * omega_si) ** 2
    weight_o = m_o * 5 * (0.3 * omega_o) ** 2
    expected = (weight_si * np.cos(omega_si * t) + weight_o
                * np.cos(omega_o * t)) / (weight_si + weight_o)
    assert np.abs(result.total_norm.mean - expected).max() < 1e-12
    only_si = dyn.vacf(tracks, max_lag_t_ps=1.0, elements=["Si"])
    assert list(only_si.c) == ["Si"]
    assert np.abs(only_si.total_norm.mean - expected).max() < 1e-12


def test_finite_difference_velocities_carry_their_attenuation_and_note():
    dt, omega = 0.02, 2 * np.pi * 3.0
    positions, _ = _circular(300, dt, omega)
    tracks = _tracks(positions, ["Si"] * 6, dt_ps=dt)
    with pytest.raises(ValueError, match="finite difference"):
        dyn.vacf(tracks)
    result = dyn.vacf(tracks, velocities="finite difference",
                      max_lag_t_ps=1.0)
    attenuation = math.sin(omega * dt) / (omega * dt)
    assert result.c["Si"].mean[0] == pytest.approx(
        (0.3 * omega * attenuation) ** 2, rel=1e-12)
    assert np.abs(result.c_norm["Si"].mean
                  - np.cos(omega * result.lag_t_ps)).max() < 1e-12
    assert any("central differences" in n for n in result.notes)
    assert result.blocks[0][0] == 1                # frame 0 has no velocity


def test_a_vibration_above_nyquist_is_folded_and_the_first_lag_is_reported():
    # regression: a 3.1 THz vibration sampled every 0.25 ps (Nyquist 2 THz)
    # gave a VDOS peak at 0.9 THz under a note that said such vibrations
    # "cannot be resolved", with no measurement of the undersampling
    dt, nu = 0.25, 3.1
    omega = 2 * np.pi * nu
    positions, velocities = _circular(400, dt, omega)
    tracks = _tracks(positions, ["Si"] * 6, dt_ps=dt, velocities=velocities,
                     box=np.eye(3) * 50.0)
    result = dyn.vacf(tracks, max_lag_t_ps=100 * dt)
    expected = math.cos(omega * dt)                # 0.1564, folded
    assert result.c_norm_first_lag["Si"] == pytest.approx(expected, abs=1e-12)
    assert result.c_norm_first_lag["total"] == pytest.approx(expected,
                                                             abs=1e-12)
    assert any("folds back" in n and "2 THz" in n for n in result.notes)
    assert any(f"Si {expected:.4g}" in n and "first lag" in n
               for n in result.notes)
    spectrum = dyn.vdos(result, window="none")
    peak = spectrum.freq_thz[int(np.argmax(spectrum.g_thz["Si"].mean))]
    assert peak == pytest.approx(abs(nu - 1.0 / dt), abs=1e-9)   # 0.9 THz
    assert any("folds back" in n for n in spectrum.notes)


def _trapezoid_by_hand(values, dt):
    total = 0.0
    for k in range(1, len(values)):
        total += 0.5 * (values[k] + values[k - 1]) * dt
    return total


def test_green_kubo_d_integrates_the_vacf_exactly():
    # circular motion: <v(0).v(t)> = (omega R)^2 cos(omega t) at every origin
    dt, omega, radius = 0.01, 2 * np.pi * 2.0, 0.3
    positions, velocities = _circular(300, dt, omega, radius=radius)
    tracks = _tracks(positions, ["Si"] * 6, dt_ps=dt, velocities=velocities)
    vac = dyn.vacf(tracks, max_lag_t_ps=1.0)
    result = dyn.green_kubo_diffusion(vac, 0.373)["Si"]
    assert result.t_last_ps == pytest.approx(0.37) and result.n_points == 38
    c = [(omega * radius) ** 2 * math.cos(omega * k * dt) for k in range(101)]
    running = [_trapezoid_by_hand(c[:k + 1], dt) / 3.0 for k in range(101)]
    assert np.abs(result.running_ang2_per_ps.mean - running).max() < 1e-12
    assert result.d_ang2_per_ps == pytest.approx(running[37], abs=1e-12)
    assert result.d_m2_per_s == pytest.approx(running[37] * 1e-8, rel=1e-12)
    assert result.c_norm_at_end == pytest.approx(math.cos(omega * 0.37),
                                                 abs=1e-12)
    assert result.blocks == vac.blocks              # for per-block pairing
    with pytest.raises(ValueError, match="beyond the last lag"):
        dyn.green_kubo_diffusion(vac, 5.0)
    with pytest.raises(ValueError, match="one string"):
        dyn.green_kubo_diffusion(vac, 0.5, elements="Si")


def test_green_kubo_d_of_ornstein_uhlenbeck_velocities():
    # v(k+1) = a v(k) + sqrt(1 - a^2) sigma xi per axis: the VACF is
    # 3 sigma^2 a^m, so D to M lags is sigma^2 dt [sum a^m - (1 + a^(M-1))/2]
    rng = np.random.default_rng(53)
    dt, tau, sigma, per_group = 0.05, 0.5, 1.0, 40
    a = math.exp(-dt / tau)
    n_atoms, n_frames = per_group * len(GROUPS), 1000
    v = np.empty((n_frames, n_atoms, 3))
    v[0] = rng.normal(0.0, sigma, size=(n_atoms, 3))
    noise = rng.normal(0.0, sigma * math.sqrt(1 - a * a),
                       size=(n_frames, n_atoms, 3))
    for k in range(1, n_frames):
        v[k] = a * v[k - 1] + noise[k]
    positions = 25.0 + np.cumsum(v, axis=0) * dt
    tracks = _tracks(positions, np.repeat(GROUPS, per_group), dt_ps=dt,
                     velocities=v, box=np.eye(3) * 100.0)
    vac = dyn.vacf(tracks, max_lag_t_ps=4.0)
    result = dyn.green_kubo_diffusion(vac, 3.0)
    lags = 61
    expected = sigma ** 2 * dt * (sum(a ** m for m in range(lags))
                                  - (1 + a ** (lags - 1)) / 2)
    values = np.array([result[g].d_ang2_per_ps for g in GROUPS])
    error = values.std(ddof=1) / math.sqrt(len(GROUPS))
    assert error < 0.05 * expected                 # the check has power
    assert abs(values.mean() - expected) < 5 * error


def test_kinetic_temperature_against_a_hand_value():
    rng = np.random.default_rng(52)
    velocities = rng.normal(size=(3, 8, 3))
    masses_amu = np.array([22.98977] * 4 + [15.9994] * 4)   # gemmi 0.7.1
    amu_kg = 1.66053906892e-27                     # CODATA 2022
    target_k = 1234.5
    for k in range(3):                             # scale each frame to 1234.5 K
        twice_ke = (masses_amu[:, None] * amu_kg * (velocities[k] * 100.0)
                    ** 2).sum()
        velocities[k] *= math.sqrt(3 * 8 * K_BOLTZMANN_J_PER_K * target_k
                                   / twice_ke)
    positions = np.zeros((3, 8, 3)) + np.arange(8)[None, :, None]
    tracks = _tracks(positions, ["Na"] * 4 + ["O"] * 4, velocities=velocities)
    temperature = dyn.kinetic_temperature(tracks)
    assert np.allclose(temperature.per_frame, target_k, rtol=1e-12)
    assert temperature.unit == "K" and temperature.n_frames == 3


# ---------------------------------------------------------------------------
# conductivity
# ---------------------------------------------------------------------------

def test_nernst_einstein_against_a_hand_computation():
    result = dyn.nernst_einstein(
        d_m2_per_s={"Na": 2.0e-9, "Ca": 5.0e-10, "O": 1.0e-12},
        charges_e={"Na": 1.0, "Ca": 2.0}, temperature_k=1500.0,
        n_atoms={"Na": 120, "Ca": 30, "O": 400}, volume_ang3=12345.6)
    volume_m3 = 12345.6e-30
    by_hand = E_CHARGE_C ** 2 / (volume_m3 * K_BOLTZMANN_J_PER_K * 1500.0) * (
        120 * 1.0 * 2.0e-9 + 30 * 4.0 * 5.0e-10)
    assert result.sigma_s_per_m == pytest.approx(by_hand, rel=1e-12)
    assert result.contributions_s_per_m["Ca"] == pytest.approx(
        E_CHARGE_C ** 2 / (volume_m3 * K_BOLTZMANN_J_PER_K * 1500.0)
        * 30 * 4.0 * 5.0e-10, rel=1e-12)
    assert any("O: a D was given but no charge" in n for n in result.notes)


def test_nernst_einstein_needs_every_input():
    good = dict(charges_e={"Na": 1.0}, temperature_k=300.0,
                n_atoms={"Na": 10}, volume_ang3=1000.0)
    with pytest.raises(TypeError):
        dyn.nernst_einstein(d_m2_per_s={"Na": 1e-9}, charges_e={"Na": 1.0},
                            n_atoms={"Na": 10}, volume_ang3=1000.0)
    with pytest.raises(ValueError, match="above 0"):
        dyn.nernst_einstein(d_m2_per_s={"Na": 1e-9},
                            **{**good, "temperature_k": 0.0})
    with pytest.raises(ValueError, match="no diffusion coefficient"):
        dyn.nernst_einstein(d_m2_per_s={"O": 1e-9}, **good)
    with pytest.raises(ValueError, match="charges_e"):
        dyn.nernst_einstein(d_m2_per_s={"Na": 1e-9},
                            **{**good, "charges_e": {}})


def test_nernst_einstein_takes_bare_numbers_only_under_a_unit_name():
    # regression: floats under 'diffusion' were read as m^2/s, so a fit's
    # d_ang2_per_ps passed by hand gave sigma 1e8 times larger, silently
    rng = np.random.default_rng(64)
    tracks = _tracks(_random_walk(rng, 60, 8, 0.2, 0.1, np.eye(3) * 20.0),
                     ["Na"] * 8)
    fit = dyn.fit_diffusion(dyn.msd(tracks), 0.5, 4.0)["Na"]
    common = dict(charges_e={"Na": 1.0}, temperature_k=1000.0,
                  n_atoms={"Na": 8}, volume_ang3=8000.0)
    with pytest.raises(ValueError, match="d_m2_per_s, whose name states its "
                                         "unit"):
        dyn.nernst_einstein({"Na": fit.d_ang2_per_ps}, **common)
    by_fit = dyn.nernst_einstein({"Na": fit}, **common)
    by_number = dyn.nernst_einstein(d_m2_per_s={"Na": fit.d_m2_per_s},
                                    **common)
    assert by_number.sigma_s_per_m == by_fit.sigma_s_per_m
    assert any(f"D used (m^2/s): Na {fit.d_m2_per_s:.6g} (MSD fit over 0.5 "
               "to 4 ps)" in n for n in by_fit.notes)
    assert any("given in d_m2_per_s" in n for n in by_number.notes)
    with pytest.raises(ValueError, match="both fits and d_m2_per_s"):
        dyn.nernst_einstein({"Na": fit}, d_m2_per_s={"Na": 1e-9}, **common)
    with pytest.raises(ValueError, match="no diffusion coefficient given"):
        dyn.nernst_einstein(**common)
    with pytest.raises(ValueError, match="n_atoms needs a map"):
        dyn.nernst_einstein({"Na": fit}, **{**common, "n_atoms": [8]})


def test_arguments_of_the_wrong_shape_are_refused_with_a_reason():
    # regression: elements='Na' was iterated letter by letter ('O' was even
    # accepted), and a flat pair raised a bare unpacking error
    rng = np.random.default_rng(65)
    tracks = _tracks(_random_walk(rng, 30, 4, 0.1, 0.1, np.eye(3) * 9.0),
                     ["Na", "Na", "O", "O"])
    result = dyn.msd(tracks)
    for symbol in ("Na", "O"):
        with pytest.raises(ValueError, match="not one string"):
            dyn.fit_diffusion(result, 0.5, 2.0, elements=symbol)
    life = dyn.bond_lifetimes(_timeline(np.ones((2, 6), dtype=bool)))
    with pytest.raises(ValueError, match=r"pair 'Na': a \(cation, anion\)"):
        dyn.residence_time(life, method="integral", pairs=("Na", "O"))
    assert list(dyn.residence_time(life, method="integral",
                                   pairs=[("Na", "O")])) == [("Na", "O")]
    # a single sentence handed over as notes was split into letters
    with pytest.raises(ValueError, match="notes needs a sequence"):
        dyn.tracks_from_arrays(["Na"], np.zeros((2, 1, 3)) + 1.0,
                               t_ps=[0.0, 0.1], box_ang=np.eye(3) * 9.0,
                               notes="made by hand")


def test_a_lag_limit_between_frames_is_noted_when_rounded():
    # regression: max_lag_t_ps = 0.14 at dt = 0.1 became 0.1 ps unnoted
    rng = np.random.default_rng(66)
    positions = _random_walk(rng, 30, 4, 0.1, 0.1, np.eye(3) * 9.0)
    tracks = _tracks(positions, ["Na"] * 4, velocities=np.ones_like(positions))
    text = "max_lag_t_ps taken to the nearest whole frame: 0.14 -> 0.1 ps"
    result = dyn.msd(tracks, max_lag_t_ps=0.14)
    assert result.lag_t_ps.tolist() == pytest.approx([0.0, 0.1])
    assert any(text in n for n in result.notes)
    assert any(text in n for n in dyn.vacf(tracks, max_lag_t_ps=0.14).notes)
    life = dyn.bond_lifetimes(_timeline(np.ones((2, 6), dtype=bool)),
                              max_lag_t_ps=0.14)
    assert any(text in n for n in life.notes)
    exact = dyn.msd(tracks, max_lag_t_ps=0.2)
    assert not any("nearest whole frame" in n for n in exact.notes)


def _ions(rng, n_frames, n_cations, n_anions, *, mode):
    box = np.eye(3) * 30.0
    start = rng.uniform(0.0, 30.0, size=(n_cations + n_anions, 3))
    walk = np.cumsum(rng.normal(0.0, 0.1, size=(n_frames, 3)), axis=0)
    positions = np.repeat(start[None], n_frames, axis=0)
    if mode == "together":                         # every cation, one walk
        positions[:, :n_cations] += walk[:, None, :]
    elif mode == "pairs":                          # each anion rides a cation
        walks = np.cumsum(rng.normal(0.0, 0.1, size=(n_frames, n_cations, 3)),
                          axis=0)
        positions[:, :n_cations] += walks
        positions[:, n_cations:] = positions[:, :n_cations] + 2.0
    elements = ["Na"] * n_cations + ["Cl"] * n_anions
    return dyn.tracks_from_arrays(elements, positions,
                                  t_ps=np.arange(n_frames) * 0.1, box_ang=box)


def test_ions_moving_as_one_give_a_haven_ratio_of_one_over_n():
    rng = np.random.default_rng(61)
    tracks = _ions(rng, 300, 8, 8, mode="together")
    charges = {"Na": 1.0, "Cl": -1.0}
    fits = dyn.fit_diffusion(dyn.msd(tracks, n_blocks=2), 0.5, 5.0)
    nernst = dyn.nernst_einstein(
        {"Na": fits["Na"]}, charges_e={"Na": 1.0}, temperature_k=1000.0,
        n_atoms=tracks.composition_model, volume_ang3=tracks.mean_volume_ang3)
    collective = dyn.collective_conductivity(
        tracks, charges_e=charges, temperature_k=1000.0, t_min_ps=0.5,
        t_max_ps=5.0, n_blocks=2)
    ratio = dyn.haven_ratio(nernst, collective)
    assert ratio.haven_ratio == pytest.approx(1 / 8, rel=1e-9)
    assert np.allclose(ratio.per_block.per_frame, 1 / 8, rtol=1e-9)
    assert collective.net_charge_e == 0.0


def test_the_haven_ratio_states_what_each_side_holds():
    # regression: an element charged only in the collective sum, a charge
    # for an element the tracks lack, and the per-block spread (of ratios,
    # not of H_R) all went unstated
    rng = np.random.default_rng(67)
    tracks = _ions(rng, 300, 8, 8, mode="pairs")
    fits = dyn.fit_diffusion(dyn.msd(tracks, n_blocks=3), 0.5, 5.0)
    nernst = dyn.nernst_einstein(
        {"Na": fits["Na"]}, charges_e={"Na": 1.0}, temperature_k=1000.0,
        n_atoms=tracks.composition_model, volume_ang3=tracks.mean_volume_ang3)
    collective = dyn.collective_conductivity(
        tracks, charges_e={"Na": 1.0, "Cl": -0.9, "F": -1.0},
        temperature_k=1000.0, t_min_ps=0.5, t_max_ps=5.0, n_blocks=3)
    assert collective.charges_e == {"Cl": -0.9, "Na": 1.0}
    assert any("charges were given for F, which the tracks do not hold" in n
               for n in collective.notes)
    ratio = dyn.haven_ratio(nernst, collective)
    assert any(n.startswith("Cl: charged in the collective sum") for n in
               ratio.notes)
    mean_of_ratios = ratio.per_block.mean
    assert mean_of_ratios != pytest.approx(ratio.haven_ratio, rel=1e-6)
    assert any(f"their mean is {mean_of_ratios:.6g} and H_R = "
               f"{ratio.haven_ratio:.6g}" in n for n in ratio.notes)
    row = ratio.as_rows()[0]
    assert row["std of per-block ratios"] == ratio.per_block.std
    assert row["mean of per-block ratios"] == mean_of_ratios
    other = dyn.nernst_einstein(
        {"Na": fits["Na"]}, charges_e={"Na": 0.8}, temperature_k=1000.0,
        n_atoms=tracks.composition_model, volume_ang3=tracks.mean_volume_ang3)
    assert any("Na (0.8 e in the Nernst-Einstein sum, 1 e in the collective "
               "one)" in n for n in dyn.haven_ratio(other, collective).notes)


def test_per_block_values_are_paired_only_over_the_same_frames():
    # regression: per-block D, sigma_NE and H_R were paired by block index
    # alone, so two blocks of 150 frames were divided by two blocks of 100
    # frames from a shorter run without a word
    rng = np.random.default_rng(68)
    box = np.eye(3) * 30.0
    positions = _random_walk(rng, 300, 16, 0.05, 0.1, box)
    elements = ["Na"] * 8 + ["Cl"] * 8
    long_run = _tracks(positions, elements, box=box)
    short_run = _tracks(positions[:200], elements, box=box)
    fits_long = dyn.fit_diffusion(dyn.msd(long_run, n_blocks=2), 0.5, 5.0)
    fits_short = dyn.fit_diffusion(dyn.msd(short_run, n_blocks=2), 0.5, 5.0)
    assert fits_long["Na"].blocks == ((0, 149), (150, 299))
    assert fits_short["Na"].blocks == ((0, 99), (100, 199))
    common = dict(temperature_k=1000.0, n_atoms=long_run.composition_model,
                  volume_ang3=long_run.mean_volume_ang3)
    nernst = dyn.nernst_einstein({"Na": fits_long["Na"]},
                                 charges_e={"Na": 1.0}, **common)
    assert nernst.per_block is not None
    assert nernst.blocks == ((0, 149), (150, 299))
    charges = {"Na": 1.0, "Cl": -1.0}
    same = dyn.collective_conductivity(long_run, charges_e=charges,
                                       temperature_k=1000.0, t_min_ps=0.5,
                                       t_max_ps=5.0, n_blocks=2)
    other = dyn.collective_conductivity(short_run, charges_e=charges,
                                        temperature_k=1000.0, t_min_ps=0.5,
                                        t_max_ps=5.0, n_blocks=2)
    assert same.blocks == nernst.blocks and other.blocks != nernst.blocks
    assert dyn.haven_ratio(nernst, same).per_block is not None
    ratio = dyn.haven_ratio(nernst, other)
    assert ratio.per_block is None
    assert any("no per-block ratio: the Nernst-Einstein blocks (frames "
               "0-149, 150-299) and the collective ones (frames 0-99, "
               "100-199) hold different frames" in n for n in ratio.notes)
    assert ratio.as_rows()[0]["mean of per-block ratios"] is None
    # two charged elements whose D come from different blocks
    mixed = dyn.nernst_einstein({"Na": fits_long["Na"],
                                 "Cl": fits_short["Cl"]}, charges_e=charges,
                                **common)
    assert mixed.per_block is None and mixed.blocks == ()
    assert any("no per-block conductivity" in n and "Cl: frames 0-99, "
               "100-199; Na: frames 0-149, 150-299" in n for n in mixed.notes)


def test_neutral_pairs_moving_together_carry_no_current():
    rng = np.random.default_rng(62)
    tracks = _ions(rng, 200, 6, 6, mode="pairs")
    collective = dyn.collective_conductivity(
        tracks, charges_e={"Na": 1.0, "Cl": -1.0}, temperature_k=1000.0,
        t_min_ps=0.5, t_max_ps=5.0)
    assert abs(collective.slope_e2ang2_per_ps) < 1e-12
    fits = dyn.fit_diffusion(dyn.msd(tracks), 0.5, 5.0)
    nernst = dyn.nernst_einstein(
        fits, charges_e={"Na": 1.0, "Cl": -1.0}, temperature_k=1000.0,
        n_atoms=tracks.composition_model, volume_ang3=tracks.mean_volume_ang3)
    assert nernst.sigma_s_per_m > 0


def test_the_collective_route_needs_every_atom_and_charge():
    rng = np.random.default_rng(63)
    positions = _random_walk(rng, 20, 4, 0.1, 0.1, np.eye(3) * 9.0)
    trajectory = _trajectory(positions, ["Na", "Na", "Cl", "Cl"],
                             np.eye(3) * 9.0, timesteps=np.arange(20),
                             unwrapped=True)
    some = dyn.collect_tracks(trajectory, timestep_fs=100.0, elements=["Na"])
    with pytest.raises(ValueError, match="every ion"):
        dyn.collective_conductivity(some, charges_e={"Na": 1.0},
                                    temperature_k=500.0, t_min_ps=0.2,
                                    t_max_ps=1.0)
    every = dyn.collect_tracks(trajectory, timestep_fs=100.0)
    with pytest.raises(ValueError, match="no charge for Cl"):
        dyn.collective_conductivity(every, charges_e={"Na": 1.0},
                                    temperature_k=500.0, t_min_ps=0.2,
                                    t_max_ps=1.0)


# ---------------------------------------------------------------------------
# bond lifetimes
# ---------------------------------------------------------------------------

def _bonds_from_history(history):
    """bulk.Bonds per frame for bond b = (cation row b, anion row n + b)."""
    n_bonds, n_frames = history.shape
    out = []
    for k in range(n_frames):
        rows = np.flatnonzero(history[:, k])
        out.append(bulk.Bonds(
            cation=rows.astype(np.int32), anion=(rows + n_bonds).astype(np.int32),
            image=np.zeros((rows.size, 3), np.int16),
            vec_ang=np.zeros((rows.size, 3)), d_ang=np.full(rows.size, 2.0),
            v_vu=np.full(rows.size, 0.2), v_bond_vu=0.075))
    elements = ["Na"] * n_bonds + ["O"] * n_bonds
    return out, elements


def _timeline(history, dt=0.1):
    bonds, elements = _bonds_from_history(history)
    return dyn.bond_timeline(bonds, elements=elements,
                             t_ps=np.arange(history.shape[1]) * dt)


def _brute_force(history, n_lags):
    """C_I and C_C by the definitions, origin by origin."""
    n_bonds, length = history.shape
    c_i, c_c = [], []
    for m in range(n_lags):
        num_i = num_c = den = 0
        for b in range(n_bonds):
            for t0 in range(length - m):
                if history[b, t0]:
                    den += 1
                    num_i += int(history[b, t0 + m])
                    num_c += int(history[b, t0:t0 + m + 1].all())
        c_i.append(num_i / den)
        c_c.append(num_c / den)
    return np.array(c_i), np.array(c_c)


def test_bond_correlations_equal_a_brute_force_count():
    rng = np.random.default_rng(71)
    history = rng.random((30, 40)) < 0.7
    result = dyn.bond_lifetimes(_timeline(history), n_blocks=2)
    for b, (start, stop) in enumerate([(0, 20), (20, 40)]):
        part = history[:, start:stop]
        c_i, c_c = _brute_force(part[part.any(axis=1)], 20)
        assert np.abs(result.intermittent[("Na", "O")].per_frame[b] - c_i
                      ).max() < 1e-15
        assert np.abs(result.continuous[("Na", "O")].per_frame[b] - c_c
                      ).max() < 1e-15


def test_a_bond_present_in_every_frame_keeps_a_correlation_of_one():
    history = np.ones((5, 30), dtype=bool)
    result = dyn.bond_lifetimes(_timeline(history))
    assert result.continuous[("Na", "O")].mean.tolist() == [1.0] * 30
    assert result.intermittent[("Na", "O")].mean.tolist() == [1.0] * 30
    assert result.n_bonds_seen[("Na", "O")] == 5


def test_gap_tolerance_bridges_short_interior_breaks_only():
    history = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    bridged = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=bool)
    result = dyn.bond_lifetimes(_timeline(history), gap_tolerance_frames=1)
    _, expected = _brute_force(bridged, 10)
    assert np.abs(result.continuous[("Na", "O")].mean - expected).max() < 1e-15
    plain = dyn.bond_lifetimes(_timeline(history))
    expected_i, _ = _brute_force(history, 10)
    assert np.abs(result.intermittent[("Na", "O")].mean - expected_i
                  ).max() < 1e-15
    assert np.abs(plain.intermittent[("Na", "O")].mean - expected_i
                  ).max() < 1e-15


def test_gap_tolerance_counts_bridged_frames_as_bonded_everywhere():
    # pins the filled-history convention the docstring and the note state:
    # h = 1 0 1 1 0 0 1 with one frame bridged gives 1, 3/4, 1/2, 1/4, 0
    history = np.array([[1, 0, 1, 1, 0, 0, 1]], dtype=bool)
    result = dyn.bond_lifetimes(_timeline(history), gap_tolerance_frames=1)
    c_c = result.continuous[("Na", "O")]
    assert c_c.mean[:5].tolist() == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert any("filled-history convention" in n for n in c_c.notes)


def test_the_intermittent_function_levels_off_at_the_uncorrelated_value():
    # regression: the integral of C_I grew linearly with t_max (2.7, 10.2,
    # 29.6 ps for 5, 20, 59 ps on a two-state chain) under the label
    # 'residence time' with no word on the level C_I keeps
    rng = np.random.default_rng(73)
    p = rng.uniform(0.2, 0.9, size=(400, 1))
    history = rng.random((400, 120)) < p             # each bond its own p_i
    result = dyn.bond_lifetimes(_timeline(history), n_blocks=2)
    level = result.intermittent_uncorrelated[("Na", "O")]
    for b, (start, stop) in enumerate([(0, 60), (60, 120)]):
        part = history[:, start:stop]
        part = part[part.any(axis=1)]
        p_i = part.mean(axis=1)
        assert level.per_frame[b] == pytest.approx(
            (p_i ** 2).sum() / p_i.sum(), rel=1e-12)
    # presence uncorrelated in time: C_I sits at that level at every lag > 0
    c_i = result.intermittent[("Na", "O")].mean
    origins = history[:, :60 - 30].sum()
    assert np.abs(c_i[1:31] - level.mean).max() < 6 * math.sqrt(
        level.mean * (1 - level.mean) / origins)
    tau = dyn.residence_time(result, method="integral", kind="intermittent",
                             t_max_ps=3.0)[("Na", "O")]
    assert tau.c_uncorrelated == pytest.approx(level.mean, rel=1e-15)
    assert any("grows with t_max_ps" in n and f"= {level.mean:.6g} here" in n
               for n in tau.notes)
    fitted = dyn.residence_time(result, method="exponential fit",
                                kind="intermittent", t_min_ps=0.1,
                                t_max_ps=1.0)[("Na", "O")]
    assert any("levels off above 0" in n for n in fitted.notes)
    continuous = dyn.residence_time(result, method="integral")[("Na", "O")]
    assert continuous.c_uncorrelated is None
    assert not any("grows with" in n for n in continuous.notes)


def test_bond_timeline_refuses_rows_that_no_bond_can_hold():
    # regression: a row of -1 silently became the last atom, and an atom in
    # both roles (an O-O 'bond') was accepted
    bonds, elements = _bonds_from_history(np.ones((2, 3), dtype=bool))
    negative = bulk.Bonds(
        cation=np.array([-1], np.int32), anion=np.array([2], np.int32),
        image=np.zeros((1, 3), np.int16), vec_ang=np.zeros((1, 3)),
        d_ang=np.array([2.0]), v_vu=np.array([0.2]), v_bond_vu=0.075)
    with pytest.raises(ValueError, match="frame 2: the bonds name a row "
                                         "below 0"):
        dyn.bond_timeline(bonds[:2] + [negative], elements=elements,
                          t_ps=[0.0, 0.1, 0.2])
    both = bulk.Bonds(
        cation=np.array([2], np.int32), anion=np.array([3], np.int32),
        image=np.zeros((1, 3), np.int16), vec_ang=np.zeros((1, 3)),
        d_ang=np.array([2.0]), v_vu=np.array([0.2]), v_bond_vu=0.075)
    with pytest.raises(ValueError, match=r"row\(s\) \[2\] \(O\) are the "
                                         "cation of one bond and the anion"):
        dyn.bond_timeline(bonds[:2] + [both], elements=elements,
                          t_ps=[0.0, 0.1, 0.2])


def test_a_bond_between_two_atoms_of_one_element_is_noted():
    # regression (the open part of the bond-row check): an O-O pair whose
    # rows appear in no other bond passed with nothing said; bond_timeline
    # has no oxidation states, but it can see that both ends are one element
    bonds, _ = _bonds_from_history(np.ones((2, 3), dtype=bool))
    elements = ["Na", "Na", "O", "O", "O", "O"]
    extra = bulk.Bonds(
        cation=np.array([4], np.int32), anion=np.array([5], np.int32),
        image=np.zeros((1, 3), np.int16), vec_ang=np.zeros((1, 3)),
        d_ang=np.array([2.0]), v_vu=np.array([0.2]), v_bond_vu=0.075)
    with_pair = [bulk.Bonds(
        cation=np.concatenate([b.cation, extra.cation]),
        anion=np.concatenate([b.anion, extra.anion]),
        image=np.concatenate([b.image, extra.image]),
        vec_ang=np.concatenate([b.vec_ang, extra.vec_ang]),
        d_ang=np.concatenate([b.d_ang, extra.d_ang]),
        v_vu=np.concatenate([b.v_vu, extra.v_vu]), v_bond_vu=0.075)
        for b in bonds]
    timeline = dyn.bond_timeline(with_pair, elements=elements,
                                 t_ps=[0.0, 0.1, 0.2])
    assert timeline.n_bonds == 3                    # kept as given
    text = "1 bonded pair(s) join two atoms of one element (O-O)"
    assert any(text in n for n in timeline.notes)
    life = dyn.bond_lifetimes(timeline)
    assert ("O", "O") in life.continuous
    assert any(text in n for n in life.notes)
    plain = dyn.bond_timeline(bonds, elements=elements, t_ps=[0.0, 0.1, 0.2])
    assert not any("one element" in n for n in plain.notes)
    with pytest.raises(ValueError, match="not one string"):
        dyn.bond_timeline(bonds, elements=elements, t_ps=[0.0, 0.1, 0.2],
                          notes="from the glass descriptors")


def test_residence_times_carry_the_notes_of_what_they_were_formed_from():
    # regression: tau lost the lifetimes' notes (threshold, time axis,
    # reader) and the function's own (a block with no such bond, NaN rows)
    history = np.zeros((2, 12), dtype=bool)
    history[:, :6] = True                           # no bond in block 1
    bonds, elements = _bonds_from_history(history)
    timeline = dyn.bond_timeline(bonds, elements=elements,
                                 t_ps=np.arange(12) * 0.1,
                                 notes=["reader: UNITS as written"])
    life = dyn.bond_lifetimes(timeline, n_blocks=2)
    for kind in dyn.BOND_KINDS:
        for args in (dict(method="integral"),
                     dict(method="exponential fit", t_min_ps=0.1,
                          t_max_ps=0.4)):
            tau = dyn.residence_time(life, kind=kind, **args)[("Na", "O")]
            assert any("reader: UNITS as written" in n for n in tau.notes)
            assert any("v > 0.075 v.u." in n for n in tau.notes)
            assert any("no such bond in block 1" in n for n in tau.notes)
            assert math.isnan(tau.per_block.per_frame[1])


def test_bonds_present_at_random_follow_p_and_p_to_the_m():
    rng = np.random.default_rng(72)
    p = 0.6
    history = rng.random((3000, 60)) < p
    result = dyn.bond_lifetimes(_timeline(history), max_lag_t_ps=0.5)
    for m in range(1, 6):
        origins = history[:, :60 - m].sum()
        for function, expected in ((result.intermittent, p),
                                   (result.continuous, p ** m)):
            error = math.sqrt(expected * (1 - expected) / origins)
            assert abs(function[("Na", "O")].mean[m] - expected) < 6 * error


def test_collect_bonds_reads_a_moving_sio4_unit_through_bulk():
    oxygen = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) \
        / math.sqrt(3) * 1.62
    centre = np.array([10.0, 10.0, 10.0])
    frame_positions = []
    for k in range(5):
        atoms = [centre] + [centre + o for o in oxygen] + [[2.0, 2.0, 2.0]]
        atoms = np.array(atoms, dtype=float)
        if k == 2:
            atoms[1] = centre + oxygen[0] / 1.62 * 3.5    # the bond breaks
        frame_positions.append(atoms)
    positions = np.array(frame_positions)
    trajectory = _trajectory(positions, ["Si", "O", "O", "O", "O", "Na"],
                             np.eye(3) * 20.0, timesteps=np.arange(5) * 10)
    ox = md_model.model_oxidation(["Si", "O", "Na"])
    timeline = dyn.collect_bonds(trajectory, ox, timestep_fs=1.0)
    assert timeline.n_bonds == 4
    assert timeline.cation.tolist() == [0, 0, 0, 0]
    assert timeline.anion.tolist() == [1, 2, 3, 4]
    assert timeline.present[0].tolist() == [True, True, False, True, True]
    assert timeline.present[1:].all()
    assert timeline.dt_ps == pytest.approx(0.01)
    result = dyn.bond_lifetimes(timeline)
    assert ("Si", "O") in result.continuous


def _exponential_lifetimes(tau, dt, n_lags):
    t = np.arange(n_lags) * dt
    rows = np.exp(-t / tau)[None].repeat(2, axis=0)
    series = Series("C_C Na-O", t, "t_ps", "ps", "1", rows, frames=[0, 1],
                    row_kind="block")
    return dyn.BondLifetimes(
        lag_t_ps=t, continuous={("Na", "O"): series},
        intermittent={("Na", "O"): series}, bonds_per_frame={},
        n_bonds_seen={("Na", "O"): 1}, gap_tolerance_frames=0,
        v_bond_vu=0.075, dt_ps=dt, block_length=n_lags,
        blocks=((0, n_lags - 1), (n_lags, 2 * n_lags - 1)))


def test_residence_time_by_integral_and_by_fit_on_an_exponential():
    tau, dt, n_lags = 5.0, 0.01, 10001
    lifetimes = _exponential_lifetimes(tau, dt, n_lags)
    fitted = dyn.residence_time(lifetimes, method="exponential fit",
                                t_min_ps=1.0, t_max_ps=20.0)[("Na", "O")]
    assert fitted.tau_ps == pytest.approx(tau, rel=1e-12)
    assert fitted.amplitude == pytest.approx(1.0, rel=1e-12)
    assert fitted.per_block.std == 0.0
    integral = dyn.residence_time(lifetimes, method="integral")[("Na", "O")]
    # the trapezoid rule on exp(-t / tau): dt/2 (1 + q)/(1 - q) (1 - q^(M-1))
    q = math.exp(-dt / tau)
    closed = dt / 2 * (1 + q) / (1 - q) * (1 - q ** (n_lags - 1))
    assert integral.tau_ps == pytest.approx(closed, rel=1e-10)
    assert abs(integral.tau_ps - tau) < 1e-5     # dt^2 / (12 tau) + tail
    with pytest.raises(ValueError, match="from t = 0"):
        dyn.residence_time(lifetimes, method="integral", t_min_ps=1.0)
    with pytest.raises(ValueError, match="needs t_min_ps"):
        dyn.residence_time(lifetimes, method="exponential fit")
    with pytest.raises(ValueError, match="not one of"):
        dyn.residence_time(lifetimes, method="fit")


# ---------------------------------------------------------------------------
# provenance, no verdicts, no Qt
# ---------------------------------------------------------------------------

def test_method_parameters_go_into_the_export_header():
    rng = np.random.default_rng(81)
    tracks = _tracks(_random_walk(rng, 30, 4, 0.1, 0.1, np.eye(3) * 9.0),
                     ["Na"] * 4)
    result = dyn.msd(tracks, n_blocks=2)
    header = Provenance(source_path=tracks.source_path, file_format="memory",
                        frames_used=tracks.frames,
                        method_parameters=result.method_parameters)
    assert any("n_blocks = 2" in line for line in header.as_lines())
    rows = result.as_rows()
    assert rows[0]["descriptor"] == "MSD Na" and "t_ps" in rows[0]


VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)


def _every_note_and_message() -> list[str]:
    rng = np.random.default_rng(91)
    positions = _random_walk(rng, 40, 6, 0.2, 0.1, np.eye(3) * 9.0)
    tracks = _tracks(positions, ["Na"] * 3 + ["Cl"] * 3)
    out = list(tracks.notes) + tracks.describe()
    result = dyn.msd(tracks, n_blocks=3)
    out += list(result.notes)
    fits = dyn.fit_diffusion(result, 0.1, 1.0)
    for fit in fits.values():
        out += list(fit.notes) + list(fit.per_block.notes)
    selfc = dyn.self_correlations(tracks, [0.15, 0.5], edges_r_ang=[0.5, 0.6],
                                  q_inv_ang=[1.0])
    out += list(selfc.notes) + list(selfc.alpha2["Na"].notes)
    out += list(selfc.van_hove["Na"][0].notes)
    vac = dyn.vacf(tracks, velocities="finite difference")
    out += list(vac.notes) + list(dyn.vdos(vac).notes)
    out += list(dyn.green_kubo_diffusion(vac, 1.0)["Na"].notes)
    out += list(dyn.collect_tracks(
        _still(5, times_ps=[0.0, 0.1, 0.2003, 0.3, 0.4]),
        frame_interval_ps=0.1).notes)
    out += list(dyn.time_axis_ps(_still(2, timesteps=[1, 2],
                                        file_format="vasp-xdatcar"),
                                 timestep_fs=1.0)[2])
    ne = dyn.nernst_einstein(fits, charges_e={"Na": 1.0, "Cl": -1.0},
                             temperature_k=900.0,
                             n_atoms=tracks.composition_model,
                             volume_ang3=tracks.mean_volume_ang3)
    co = dyn.collective_conductivity(tracks, charges_e={"Na": 1.0, "Cl": -0.5},
                                     temperature_k=900.0, t_min_ps=0.1,
                                     t_max_ps=1.0, n_blocks=2)
    out += list(ne.notes) + list(co.notes)
    out += list(dyn.haven_ratio(ne, co).notes)
    ne_na = dyn.nernst_einstein({"Na": fits["Na"]}, charges_e={"Na": 2.0},
                                temperature_k=900.0,
                                n_atoms=tracks.composition_model,
                                volume_ang3=tracks.mean_volume_ang3)
    out += list(dyn.haven_ratio(ne_na, co).notes)
    history = np.random.default_rng(92).random((4, 12)) < 0.5
    life = dyn.bond_lifetimes(_timeline(history), gap_tolerance_frames=1)
    out += list(life.notes) + list(life.continuous[("Na", "O")].notes)
    out += list(life.intermittent[("Na", "O")].notes)
    out += list(dyn.residence_time(life, method="integral")[("Na", "O")].notes)
    for kind_args in (dict(method="integral"),
                      dict(method="exponential fit", t_min_ps=0.1,
                           t_max_ps=0.5)):
        out += list(dyn.residence_time(life, kind="intermittent",
                                       **kind_args)[("Na", "O")].notes)
    wrapped = np.array([[[8.9, 1.0, 1.0]], [[0.1, 1.0, 1.0]]])   # a crossing
    reset = positions.copy() + 18.0
    reset[20:] = np.mod(reset[20:], 9.0)
    for call in (
            lambda: dyn.tracks_from_arrays(["Na"], wrapped, t_ps=[0.0, 0.1],
                                           box_ang=np.eye(3) * 9.0),
            lambda: dyn.collect_tracks(
                _trajectory(positions + 18.0, ["Na"] * 6, np.eye(3) * 9.0,
                            timesteps=np.arange(40), unwrapped=reset),
                timestep_fs=100.0),
            lambda: dyn.collect_tracks(
                _still(4, timesteps=[0, 1, 2, 4]), frame_interval_ps=1.0),
            lambda: dyn.collect_tracks(
                _still(4, timesteps=[0, 1, 2, 1]), frame_interval_ps=1.0),
            lambda: dyn.collect_tracks(
                _still(4, times_ps=[0.0, 1.0, 3.0, 4.0]),
                frame_interval_ps=1.0),
            lambda: dyn.collect_tracks(
                _still(4, timesteps=[0, 1, 2, 3],
                       times_ps=[0.0, 1.0, 2.6, 3.0])),
            lambda: dyn.nernst_einstein({"Na": 0.2}, charges_e={"Na": 1},
                                        temperature_k=1.0, n_atoms={"Na": 1},
                                        volume_ang3=1.0),
            lambda: dyn.bond_timeline(
                _bonds_from_history(np.ones((1, 2), dtype=bool))[0]
                + [bulk.Bonds(cation=np.array([1], np.int32),
                              anion=np.array([0], np.int32),
                              image=np.zeros((1, 3), np.int16),
                              vec_ang=np.zeros((1, 3)), d_ang=np.ones(1),
                              v_vu=np.ones(1), v_bond_vu=0.075)],
                elements=["Na", "O"], t_ps=[0.0, 0.1, 0.2]),
            lambda: dyn.residence_time(life, method="integral",
                                       pairs=("Na", "O")),
            lambda: dyn.fit_diffusion(result, 0.1, 1.0, elements="Na"),
            lambda: dyn.green_kubo_diffusion(vac, 50.0)):
        try:
            call()
        except ValueError as error:
            out.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    for call in (
            lambda: dyn.collect_tracks(_still(3)),
            lambda: dyn.collect_tracks(_still(4, timesteps=[0, 1, 2, 5]),
                                       timestep_fs=1.0),
            lambda: dyn.vacf(tracks),
            lambda: dyn.msd(tracks, n_blocks=40),
            lambda: dyn.fit_diffusion(result, 0.1, 50.0),
            lambda: dyn.self_correlations(tracks, [0.0]),
            lambda: dyn.nernst_einstein(d_m2_per_s={"Na": -1.0},
                                        charges_e={"Na": 1},
                                        temperature_k=1.0, n_atoms={"Na": 1},
                                        volume_ang3=1.0),
            lambda: dyn.collective_conductivity(
                tracks, charges_e={"Na": 1.0}, temperature_k=1.0,
                t_min_ps=0.1, t_max_ps=1.0),
            lambda: dyn.residence_time(life, method="exponential fit")):
        try:
            call()
        except ValueError as error:
            out.append(str(error))
        else:
            raise AssertionError("expected a refusal")
    return out


def test_no_note_or_message_carries_a_verdict():
    notes = _every_note_and_message()
    assert len(notes) >= 40
    assert [n for n in notes if VERDICT.search(n)] == []


def test_no_string_in_the_module_carries_a_verdict():
    source = (ROOT / "facet" / "core" / "md_dynamics.py").read_text(
        encoding="utf-8")
    strings = [node.value for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(strings) > 100
    assert [s for s in strings if VERDICT.search(s)] == []


def test_the_module_imports_without_qt():
    code = ("import sys, facet.core.md_dynamics;"
            "print('QT' if any(m.startswith(('PySide6', 'matplotlib')) "
            "for m in sys.modules) else 'CLEAN')")
    result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout
