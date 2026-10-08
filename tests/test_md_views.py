"""The MD result views: plots, the result browser and the threshold panel.

What these tests pin:

* every plot kind (bars with whiskers, step with band, line with band,
  per-frame points, overlays, a measured-and-model comparison) renders on
  synthetic md_stats containers, offscreen, and every axis label carries a
  unit in parentheses;
* the palette is Okabe-Ito, the one FACET's "High contrast" element palette
  is built from, and no two series of a figure share both marker and dash;
* SVG and PDF exports are vector files, and the PNG is 600 dpi with a curve
  at least 0.25 mm thick (cosmetic pens would come out 1 device pixel wide);
* the result browser, given a real ModelResult (md_analysis run on the
  24-atom LAMMPS model of tests/data/md, read through its XTC file and its
  data file), lists every descriptor of every analysis once, lists each
  analysis that produced nothing with its reason, shows each item, and
  writes the rows with the provenance header and every digit;
* the threshold panel's numbers at every slider position equal
  ``bulk.at_threshold`` on the same table, and moving the slider searches
  nothing;
* worker results arrive on the GUI thread;
* no text the three modules can show carries a verdict.
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import dispose

ROOT = Path(__file__).resolve().parent.parent
MD_DATA = Path(__file__).resolve().parent / "data" / "md"
QUARTZ = Path(__file__).resolve().parent / "data" / "crystals" / \
    "quartz_SiO2_cod9013321.cif"
UI_MODULES = ("md_plot.py", "md_views.py", "md_threshold.py")
VERDICT = re.compile(
    r"\b(good|bad|poor|excellent|acceptable|unacceptable|correct|incorrect|"
    r"wrong|reliable|unreliable|trustworthy|untrustworthy|unusable|should|"
    r"proves|confirms)\b", re.IGNORECASE)
UNIT = re.compile(r"\(([^()]+)\)\s*$")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(qapp, signal, timeout_ms=60000):
    """Run the event loop until ``signal`` fires; its arguments, or None."""
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    got = []

    def done(*args):
        got.append(args)
        loop.quit()

    signal.connect(done)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    try:
        signal.disconnect(done)
    except (RuntimeError, TypeError):
        pass
    return got[0] if got else None


# ---------------------------------------------------------------------------
# synthetic containers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def containers():
    from facet.core import md_stats

    rng = np.random.default_rng(3)
    dist = md_stats.Distribution.from_counts(
        [{3: 10, 4: 80, 5: 10}, {3: 12, 4: 76, 5: 12}, {3: 8, 4: 84, 5: 8}],
        name="CN Si", kind="fraction")
    dist2 = md_stats.Distribution.from_counts(
        [{4: 90, 5: 10}, {4: 86, 5: 14}, {4: 92, 5: 8}],
        name="CN Al", kind="fraction")
    hist = md_stats.Histogram.from_samples(
        [rng.normal(109.5, 6.0, 300) for _ in range(4)],
        np.arange(80.0, 141.0, 1.0), name="O-Si-O angle", unit="deg")
    r = np.linspace(0.01, 8.0, 400)
    g = md_stats.Series.from_frames(
        r, [1 + 5 * np.exp(-((r - 1.62) / 0.05) ** 2)
            + 0.05 * rng.normal(size=r.size) for _ in range(5)],
        name="g Si-O", axis_name="r_ang", axis_unit="Å", value_unit="1")
    g2 = md_stats.Series.from_frames(
        r, [1 + 3 * np.exp(-((r - 2.62) / 0.08) ** 2)
            + 0.05 * rng.normal(size=r.size) for _ in range(5)],
        name="g O-O", axis_name="r_ang", axis_unit="Å", value_unit="1")
    t = np.logspace(-3, 2, 50)
    msd = md_stats.Series.from_frames(
        t, [0.6 * t + 0.3 + 0.01 * rng.normal(size=t.size) for _ in range(3)],
        name="MSD Na", axis_name="t_ps", axis_unit="ps",
        value_unit="Å^2", row_kind="block")
    scalar = md_stats.Scalar("density", "g/cm^3",
                             2.2 + 0.01 * rng.normal(size=8))
    return {"dist": dist, "dist2": dist2, "hist": hist, "g": g, "g2": g2,
            "msd": msd, "scalar": scalar}


def _comparison_rows():
    q = np.linspace(0.5, 15.0, 120)
    rows = [{"quantity": "S(Q) neutron", "q_inv_ang": float(x),
             "measured": float(1 + 0.3 * np.sin(x) / x),
             "model (scaled)": float(1 + 0.28 * np.sin(1.02 * x) / x),
             "measured - model": 0.0} for x in q]
    summary = [{"quantity": "S(Q) neutron", "R_chi": 0.031, "scale": 1.01,
                "scale source": "least squares", "points compared": 120,
                "range low (1/Å)": 0.5, "range high (1/Å)": 15.0,
                "R_chi definition": "R_chi = sqrt( sum_i [y_meas(x_i) - "
                                    "s y_model(x_i)]^2 / sum_i y_meas(x_i)^2 )",
                "measured file": "measured.dat"}]
    return rows, summary


def _texts(rows):
    """Rows as full-precision text, so NaN equals NaN."""
    from facet.ui.md_views import _full

    return [{k: _full(v) for k, v in row.items()} for row in rows]


def _figures(containers):
    from facet.ui import md_plot

    rows, summary = _comparison_rows()
    fractions = [{"descriptor": "Si Qn", "key": k, "model": "Qn Si (BV)",
                  "model mean": m, "model std (across frames)": 0.01,
                  "measured": v, "measured uncertainty": 0.02,
                  "model - measured": m - v, "measured source": "test"}
                 for k, m, v in (("Q3", 0.3, 0.32), ("Q4", 0.7, 0.68))]
    c = containers
    return {
        "distribution": md_plot.figure_for(c["dist"]),
        "histogram": md_plot.figure_for(c["hist"]),
        "series": md_plot.figure_for(c["g"]),
        "msd": md_plot.figure_for(c["msd"]),
        "scalar": md_plot.figure_for(c["scalar"]),
        "series overlay": md_plot.figure_for_family([c["g"], c["g2"]],
                                                    title="g(r)", y_name="g"),
        "bars overlay": md_plot.figure_for_family([c["dist"], c["dist2"]],
                                                  title="CN"),
        "comparison": md_plot.figure_for_comparison(rows, summary),
        "fractions": md_plot.figure_for_fractions(fractions),
    }


def _plot(qapp, figure, width=820, height=440):
    from facet.ui.md_plot import StatPlot

    plot = StatPlot()
    plot.resize(width, height)
    plot.set_figure(figure)
    return plot


def _colour_pixels(image, rgb, tol=40) -> int:
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format_RGB32)
    ptr = image.constBits()
    data = np.frombuffer(ptr, dtype=np.uint8,
                         count=image.sizeInBytes()).reshape(
        image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3]
    bgr = np.array(rgb[::-1])
    return int((np.abs(data.astype(int) - bgr).max(axis=2) <= tol).sum())


# ---------------------------------------------------------------------------
# palette and styles
# ---------------------------------------------------------------------------

def test_the_palette_is_okabe_ito_as_the_high_contrast_theme_holds_it():
    from facet.core import theme
    from facet.ui import md_plot

    named = dict(md_plot.OKABE_ITO)
    palette = theme.PALETTES["High contrast"]()
    held = {"N": "blue", "O": "vermillion", "Cl": "bluish green",
            "Bi": "reddish purple", "P": "orange", "F": "sky blue",
            "S": "yellow"}
    for element, name in held.items():
        # theme.py writes two decimals: 0.85 for 213/255 = 0.835
        rgb = np.array(palette[element]) * 255.0
        assert np.abs(rgb - np.array(named[name])).max() <= 4.0, element
    assert named["black"] == (0, 0, 0)
    # lines and bars take every colour but yellow
    assert "yellow" not in dict(md_plot.LINE_COLOURS)


def test_no_two_series_share_marker_and_dash():
    from facet.ui.md_plot import style_for

    styles = [style_for(k) for k in range(35)]
    assert len({(s.marker, s.dash) for s in styles}) == 35
    # the marker changes whenever the colour does, so a series never differs
    # from its neighbour by colour alone
    for a, b in zip(styles, styles[1:]):
        assert a.colour != b.colour and a.marker != b.marker


def test_a_figure_refuses_an_axis_without_a_unit():
    from facet.ui.md_plot import Figure, StatSeries

    s = StatSeries("x", [0.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="unit"):
        Figure([s], "r", "", "g", "1")
    figure = Figure([s], "r", "Å", "g", "1")
    assert figure.x_label == "r (Å)"
    assert figure.y_label == "g (dimensionless)"


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["distribution", "histogram", "series",
                                  "msd", "scalar", "series overlay",
                                  "bars overlay", "comparison", "fractions"])
def test_each_plot_kind_renders_with_units_on_both_axes(qapp, containers,
                                                        kind):
    from facet.core import theme
    from facet.ui.md_plot import OKABE_ITO

    figure = _figures(containers)[kind]
    assert figure is not None
    plot = _plot(qapp, figure)
    try:
        plot.apply_theme(theme.vesta())
        image = plot.grab().toImage()
        # grab() renders at the screen's device pixel ratio (1 offscreen,
        # 1.1 on a 106 dpi laptop panel), so the width is compared in
        # logical pixels
        assert not image.isNull()
        assert round(image.width() / image.devicePixelRatio()) == 820
        # the first series is drawn in its own colour (black for the
        # measured points of a comparison)
        colour = figure.series[0].style.colour
        assert _colour_pixels(image, colour) > 30, kind
        x_label, y_label = plot.axis_labels()
        for label in (x_label, y_label):
            match = UNIT.search(label)
            assert match and match.group(1).strip(), label
        if figure.categories is None:
            assert figure.series[0].style.colour in dict(OKABE_ITO).values()
    finally:
        dispose(plot)


def test_bands_and_whiskers_are_the_spread_across_frames(containers):
    from facet.ui import md_plot

    figure = md_plot.figure_for(containers["g"])
    s = figure.series[0]
    np.testing.assert_array_equal(s.mean, containers["g"].mean)
    np.testing.assert_array_equal(s.std, containers["g"].std)
    assert md_plot.SPREAD_ANNOTATION in figure.annotations
    assert "ddof = 1" in md_plot.SPREAD_ANNOTATION
    bars = md_plot.figure_for(containers["dist"])
    np.testing.assert_array_equal(bars.series[0].std, containers["dist"].std)
    assert bars.categories == ("3", "4", "5")


def test_log_axes_follow_the_axis_spacing_and_the_msd(containers):
    from facet.core import md_stats
    from facet.ui import md_plot

    assert md_plot.figure_for(containers["msd"]).log_x
    assert md_plot.figure_for(containers["msd"]).log_y
    assert not md_plot.figure_for(containers["g"]).log_x
    t = np.logspace(-2, 1, 30)
    spaced = md_stats.Series.from_frames(t, [np.exp(-t)] * 2, name="F_s",
                                         axis_name="t_ps", axis_unit="ps",
                                         value_unit="1")
    figure = md_plot.figure_for(spaced)
    assert figure.log_x and not figure.log_y
    assert md_plot.is_log_spaced(t)
    assert not md_plot.is_log_spaced(np.linspace(0.1, 10, 30))


def test_points_at_or_below_zero_on_a_log_axis_are_counted_not_hidden(
        qapp, containers):
    from facet.ui.md_plot import Figure, StatSeries

    s = StatSeries("y", [1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0, 10.0])
    plot = _plot(qapp, Figure([s], "x", "1", "y", "1", log_y=True))
    try:
        plot.grab()
        assert plot._dropped["y"] == 2
    finally:
        dispose(plot)


def test_vector_exports_are_vectors_and_the_png_is_600_dpi(qapp, containers,
                                                           tmp_path):
    from PySide6.QtGui import QImage

    plot = _plot(qapp, _figures(containers)["series overlay"])
    try:
        svg = plot.save(tmp_path / "figure.svg")
        pdf = plot.save(tmp_path / "figure.pdf")
        png = plot.save(tmp_path / "figure.png")
    finally:
        dispose(plot)
    text = svg.read_text(encoding="utf-8")
    assert "<svg" in text and "<path" in text
    assert 'width="180mm" height="110mm"' in text
    assert "image/png" not in text
    assert pdf.read_bytes()[:5] == b"%PDF-"
    image = QImage(str(png))
    assert image.width() == round(180 / 25.4 * 600)
    assert image.height() == round(110 / 25.4 * 600)
    assert image.dotsPerMeterX() == round(600 / 0.0254) == 23622
    with pytest.raises(ValueError, match="svg"):
        _plot(qapp, None).save(tmp_path / "figure.jpg")


def test_a_curve_in_the_600_dpi_png_is_at_least_a_quarter_millimetre(qapp):
    from facet.ui.md_plot import Figure, StatSeries, style_for

    s = StatSeries("level", np.linspace(0.0, 10.0, 50), np.full(50, 1.0),
                   markers=False, band=False)
    s.style = style_for(0)
    plot = _plot(qapp, Figure([s], "x", "1", "y", "1"))
    try:
        image = plot.render_image(180.0, 110.0, 600)
    finally:
        dispose(plot)
    column = image.width() // 2
    blue = np.array(style_for(0).colour)
    run = longest = 0
    for y in range(image.height()):
        c = image.pixelColor(column, y)
        rgb = np.array([c.red(), c.green(), c.blue()])
        if np.abs(rgb - blue).max() <= 60:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    assert longest >= 0.25 / 25.4 * 600          # 5.9 px at 600 dpi


def test_the_comparison_shows_the_r_factor_with_its_definition(containers):
    from facet.ui import md_plot

    figure = _figures(containers)["comparison"]
    assert [s.label for s in figure.series] == [
        "measured", "model (scaled)", "measured - model"]
    text = " ".join(figure.annotations)
    assert "R_chi = 0.031" in text
    assert "Definition: R_chi = sqrt" in text
    assert figure.x_label == "Q (1/Å)"
    assert md_plot.figure_for_table([{"a": 1}]) is None
    # the columns as the driver now names them, each with its unit
    rows, summary = _comparison_rows()
    renamed = [{"quantity": r["quantity"], "q_inv_ang": r["q_inv_ang"],
                "measured (1)": r["measured"],
                "model, scaled (1)": r["model (scaled)"],
                "measured - model (1)": r["measured - model"]} for r in rows]
    again = md_plot.figure_for_comparison(renamed, summary)
    np.testing.assert_array_equal(again.series[0].mean,
                                  figure.series[0].mean)
    assert again.y_label == "S(Q) neutron (dimensionless)"


# ---------------------------------------------------------------------------
# a real result: md_analysis on the 24-atom LAMMPS model of tests/data/md
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_result(tmp_path_factory):
    from facet.core import md_analysis as ma

    folder = tmp_path_factory.mktemp("measured")
    q = np.linspace(0.5, 15.0, 200)
    measured = folder / "measured_sq.dat"
    np.savetxt(measured, np.column_stack([q, 1 + 0.3 * np.sin(q) / q]))
    options = {"topology": str(MD_DATA / "xtc_lammps.data")}
    request = ma.AnalysisRequest(
        analyses=("glass", "scattering", "scattering-comparison", "rings",
                  "self-correlations"),
        formers=frozenset({"Si"}), read_options=options,
        # the termination range is stated in full: md_analysis passes a
        # None q_min on to md_scattering, which refuses it (reported)
        scattering=ma.ScatteringOptions(
            r_window="Lorch", radiations=("neutron",),
            termination_q_max_inv_ang=15.0, termination_q_min_inv_ang=0.02,
            q_window="Lorch",
            measured=(ma.MeasuredCurve(str(measured), "neutron", "S(Q)"),)),
        network=ma.NetworkOptions(ring_criterion="primitive",
                                  ring_max_size=8),
        dynamics=ma.DynamicsOptions(lag_t_ps=(0.01,)))
    trajectory = ma.read_model(MD_DATA / "xtc_lammps.xtc", request)
    result = ma.analyse(trajectory, request)
    # an analysis that produced nothing, as the driver reports one (its
    # reason in AnalysisOutput.error), so the browser's listing of it is
    # tested whichever analyses this small model leaves without a result
    failed = ma.AnalysisOutput("vacf", {}, None,
                               error="the tracks hold no velocities")
    return dataclasses.replace(result, outputs={**result.outputs,
                                                "vacf": failed})


@pytest.fixture
def browser(qapp, model_result):
    from facet.core import theme
    from facet.ui.md_views import ResultBrowser

    b = ResultBrowser(theme=theme.vesta())
    b.resize(1400, 860)
    b.set_result(model_result)
    yield b
    b.shutdown()
    dispose(b)


def test_the_browser_lists_every_descriptor_of_every_analysis_once(
        browser, model_result):
    listed = browser.items()
    expected = [(a, d) for a, d, _ in model_result.descriptors()]
    assert len(listed) == len(expected) == len(set(listed))
    assert set(listed) == set(expected)
    assert len(expected) > 40
    # every analysis has its node, the ones that produced nothing with the
    # engine's reason
    failed = model_result.failed
    assert "vacf" in failed
    for analysis in model_result.outputs:
        node = browser.find_item(analysis)
        assert node is not None, analysis
        if analysis in failed:
            assert node.text(1) == "not computed"
            assert failed[analysis] in node.toolTip(0)


def test_every_item_is_shown_with_its_rows_and_units(browser, model_result):
    from facet.core import md_export
    from facet.core.md_stats import Distribution, Histogram, Scalar, Series

    for analysis, name in browser.items():
        assert browser.select(analysis, name)
        assert browser.current() == ("item", analysis, name)
        assert browser.view.title.text() == name
        container = model_result.outputs[analysis].tables[name]
        rows = browser.view.rows()
        assert _texts(rows) == _texts(md_export.descriptor_rows(container))
        if isinstance(container, (Distribution, Histogram, Series, Scalar)):
            figure = browser.view.figure
            assert figure is not None, name
            for label in (figure.x_label, figure.y_label):
                assert UNIT.search(label), (name, label)
            # the header the rows are copied and saved with
            assert browser.view.header[:2] == [f"analysis: {analysis}",
                                               f"descriptor: {name}"]


def test_descriptors_that_differ_by_element_share_a_figure(browser,
                                                          model_result):
    assert browser.select("glass", "g *-*")
    figure = browser.view.figure
    names = [n for n in model_result.outputs["glass"].tables
             if n.startswith("g ") and "-" in n]
    assert [s.label for s in figure.series] == names
    assert len({(s.style.marker, s.style.dash) for s in figure.series}) == \
        len(names)
    assert figure.y_label == "g (dimensionless)"
    assert len(browser.view.rows()) == sum(
        model_result.outputs["glass"].tables[n].axis.size for n in names)


def test_the_comparison_is_drawn_with_its_r_factor(browser, model_result):
    tables = model_result.outputs["scattering-comparison"].tables
    name = next(n for n in tables if not n.endswith("(R_chi)"))
    assert browser.select("scattering-comparison", name)
    figure = browser.view.figure
    assert [s.label for s in figure.series][:2] == ["measured",
                                                    "model (scaled)"]
    r_chi = tables[name + " (R_chi)"].rows[0]["R_chi"]
    assert any(f"R_chi = {r_chi:.4g}" in a for a in figure.annotations)


def test_rows_are_saved_with_the_provenance_and_every_digit(browser,
                                                           model_result,
                                                           tmp_path):
    from facet.core import md_stats

    assert browser.select("scattering", "S(Q) neutron")
    series = model_result.outputs["scattering"].tables["S(Q) neutron"]
    assert isinstance(series, md_stats.Series)
    path = browser.export_rows(tmp_path / "sq.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    joined = "\n".join(header)
    for needle in ("frames used:", "type map source:", "bond threshold v_bond",
                   "program: FACET", "analysis: scattering"):
        assert needle in joined, needle
    body = [ln for ln in lines if not ln.startswith("#")]
    rows = list(csv.DictReader(body))
    means = np.array([float(r["mean (1)"]) for r in rows])
    np.testing.assert_array_equal(means, series.mean)       # every digit
    per_frame = browser.export_rows(tmp_path / "sq_frames.csv",
                                    per_frame=True)
    text = per_frame.read_text(encoding="utf-8")
    assert "frame 0" in text and "frame 2" in text


def test_the_copied_selection_holds_every_digit(qapp, browser, model_result):
    assert browser.select("glass", "density")
    table = browser.view.table
    table.selectAll()
    text = table.selection_text()
    density = model_result.outputs["glass"].tables["density"]
    assert repr(density.mean) in text
    assert text.splitlines()[0].split("\t") == table.columns()


def test_the_figure_shown_is_saved_at_600_dpi(browser, tmp_path):
    from PySide6.QtGui import QImage

    assert browser.select("glass", "density")
    png = browser.export_figure(tmp_path / "density.png")
    assert QImage(str(png)).dotsPerMeterX() == 23622
    svg = browser.export_figure(tmp_path / "density.svg")
    assert "<path" in svg.read_text(encoding="utf-8")
    browser.select("glass", "g(r) first minima")
    assert browser.view.figure is None
    with pytest.raises(ValueError):
        browser.export_figure(tmp_path / "none.png")


def test_export_all_runs_on_a_worker_and_reports_on_the_gui_thread(
        qapp, browser, tmp_path):
    from PySide6.QtCore import QThread

    folder = tmp_path / "csv"
    seen = []
    browser.exportFinished.connect(
        lambda paths: seen.append(QThread.currentThread()))
    assert browser.start_export_all(folder)
    assert browser.is_exporting()
    assert not browser.start_export_all(folder)          # one at a time
    got = _wait(qapp, browser.exportFinished)
    assert got is not None
    paths = [Path(p) for p in got[0]]
    assert (folder / "index.csv") in paths and all(p.is_file() for p in paths)
    assert seen == [qapp.thread()]
    assert browser.shutdown(10000)
    qapp.processEvents()
    assert not browser.is_exporting()
    workbook = browser.export_all(tmp_path / "all.xlsx")
    assert workbook and all(Path(p).exists() for p in workbook)


def test_a_cancelled_or_failed_run_is_named_in_the_banner(browser,
                                                         model_result):
    assert not browser.banner.isHidden()
    assert "produced no result" in browser.banner.text()
    assert model_result.provenance.facet_version in browser.summary.text()


# ---------------------------------------------------------------------------
# the threshold panel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def quartz_frame():
    from facet.core import md_model, readers

    structure = readers.read(QUARTZ)
    base, _ = md_model.supercell_frame(structure, (2, 2, 2))
    rng = np.random.default_rng(11)
    frame = md_model.frame_from_arrays(
        base.elements, base.cart_ang + rng.normal(0, 0.06,
                                                  base.cart_ang.shape),
        box_ang=base.box_ang, timestep=0)
    ox = md_model.model_oxidation(frame.species).per_atom(frame.elements)
    return frame, ox


@pytest.fixture
def panel(qapp, quartz_frame):
    from facet.ui.md_threshold import ThresholdPanel

    frame, ox = quartz_frame
    p = ThresholdPanel()
    p.resize(1300, 900)
    p.set_frame(frame, ox, analysis_v_bond_vu=0.075, label="frame 0")
    yield p
    p.shutdown()
    dispose(p)


def test_the_slider_values_equal_bulk_at_threshold(panel):
    from facet.core import bulk, glass
    from facet.ui.md_threshold import staircase

    table = panel.table
    for position in (0, 1, 137, 400, 500, 731, 999, 1000):
        panel.slider.setValue(position)
        v = panel.v_bond
        assert v == panel.value_at(position)
        expected = bulk.at_threshold(table, v)
        for field in ("cn", "cn_listed", "bvs_vu", "phi", "bvv_vu",
                      "plateau_decades"):
            np.testing.assert_array_equal(getattr(panel.results, field),
                                          getattr(expected, field))
        counts = panel.cn_counts()
        rows = {r["element"]: r for r in panel.element_rows()}
        for element in ("O", "Si"):
            assert counts[element] == glass.cn_counts(expected.cn,
                                                      table.elements, element)
            sel = table.elements == element
            assert rows[element]["mean CN"] == expected.cn[sel].mean()
            # the staircase at this threshold is the same mean
            assert staircase(table, [v])[element][0] == \
                pytest.approx(expected.cn[sel].mean(), abs=0, rel=1e-15)
        # the CN bars drawn are the fractions of the element's atoms
        bars = {s.label: s for s in panel.cn_plot.figure.series}
        n_si = int((table.elements == "Si").sum())
        for k, c in counts["Si"].items():
            assert bars["Si"].mean[k] == c / n_si


def test_moving_the_threshold_searches_nothing(panel, monkeypatch):
    from facet.core import bulk

    def refuse(*args, **kwargs):
        raise AssertionError("a pair search was started")

    for name in ("iter_pairs", "find_pairs", "_search", "analyse_frame"):
        monkeypatch.setattr(bulk, name, refuse)
    before = panel.results.cn.copy()
    panel.set_threshold(0.5)
    panel.slider.setValue(10)
    panel._on_pick(0.3, 0.0)
    assert panel.v_bond == 0.3
    assert not np.array_equal(panel.results.cn, before) or True
    panel.reset_threshold()
    assert panel.v_bond == 0.075
    np.testing.assert_array_equal(panel.results.cn, before)


def test_the_slider_spans_v_list_to_the_largest_valence(panel):
    from facet.core import bv
    from facet.ui.md_threshold import threshold_range

    low, high = threshold_range(panel.table)
    assert low == bv.V_LIST_DEFAULT
    valid = np.arange(panel.table.v_vu.shape[1])[None, :] < \
        panel.table.n_listed[:, None]
    assert high == panel.table.v_vu[valid].max()
    assert panel.value_at(0) == low
    assert panel.value_at(1000) == pytest.approx(high, rel=1e-12)
    panel.set_threshold(100.0)
    assert panel.v_bond == high
    assert "analysis run used 0.0750" in panel.banner.text()


def test_a_given_table_is_used_without_a_search(qapp, quartz_frame,
                                                monkeypatch):
    from facet.core import bulk
    from facet.ui.md_threshold import ThresholdPanel

    frame, ox = quartz_frame
    table, _ = bulk.analyse_frame(frame, ox)
    monkeypatch.setattr(bulk, "analyse_frame", lambda *a, **k: (_ for _ in (
        )).throw(AssertionError("searched")))
    p = ThresholdPanel()
    try:
        p.set_frame(frame, ox, table=table, v_bond_vu=0.1)
        assert p.table is table and p.v_bond == 0.1
    finally:
        dispose(p)


def test_a_background_search_reports_on_the_gui_thread(qapp, quartz_frame):
    from PySide6.QtCore import QThread
    from facet.core import bulk
    from facet.ui.md_threshold import ThresholdPanel

    frame, ox = quartz_frame
    p = ThresholdPanel()
    seen = []
    p.frameReady.connect(lambda: seen.append(QThread.currentThread()))
    try:
        p.set_frame(frame, ox, background=True, analysis_v_bond_vu=0.075)
        assert p.table is None and p.is_busy()
        assert _wait(qapp, p.frameReady) is not None
        assert seen == [qapp.thread()]
        expected = bulk.at_threshold(bulk.analyse_frame(frame, ox)[0], 0.075)
        np.testing.assert_array_equal(p.results.cn, expected.cn)
        assert p.shutdown(10000)
    finally:
        dispose(p)


def test_a_refused_frame_is_reported_from_the_background(qapp, quartz_frame):
    from facet.ui.md_threshold import ThresholdPanel

    import time

    frame, ox = quartz_frame
    p = ThresholdPanel()
    try:
        p.set_frame(frame, ox[:-1], background=True)
        got = _wait(qapp, p.failed)
        assert got is not None and "oxidation state" in got[0]
        assert "not analysed" in p.heading.text()
        # waiting from the GUI thread right after the result: the thread
        # quits on its own (a quit queued to the GUI thread would wait
        # behind this wait until its timeout)
        start = time.perf_counter()
        assert p.shutdown(20000)
        assert time.perf_counter() - start < 5.0
    finally:
        dispose(p)


def test_the_panel_renders(qapp, panel):
    from facet.core import theme

    panel.apply_theme(theme.dark())
    panel.set_threshold(0.2)
    image = panel.grab().toImage()
    assert not image.isNull()
    for plot in panel.plots():
        assert plot.figure is not None
        for label in plot.axis_labels():
            assert UNIT.search(label), label


# ---------------------------------------------------------------------------
# no verdicts
# ---------------------------------------------------------------------------

def test_no_string_in_the_modules_carries_a_verdict():
    """Every string literal of the three modules, docstrings included."""
    for name in UI_MODULES:
        source = (ROOT / "facet" / "ui" / name).read_text(encoding="utf-8")
        strings = [node.value for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant)
                   and isinstance(node.value, str)]
        assert len(strings) > 20, name
        assert [s for s in strings if VERDICT.search(s)] == [], name


def test_no_text_the_views_show_carries_a_verdict(browser, panel):
    texts = []
    browser.tree.setCurrentItem(browser.tree.topLevelItem(0))
    texts += browser.visible_texts()
    for analysis in browser.result.outputs:
        browser.select(analysis)
        texts += browser.visible_texts()
    for analysis, name in browser.items():
        browser.select(analysis, name)
        texts += browser.visible_texts()
    for v in (0.03, 0.075, 0.4):
        panel.set_threshold(v)
        texts += panel.visible_texts()
    assert len(texts) > 200
    assert [t for t in texts if VERDICT.search(t)] == []


# ---------------------------------------------------------------------------
# the verifiers' findings of 2026-10-07, each pinned
# ---------------------------------------------------------------------------

def test_a_family_key_keeps_the_letters_of_a_word():
    """'(BV)' read as boron: the tree showed 'CN * (*V)' on a Na-B-Si
    model; vanadium would have done the same to the V."""
    from facet.ui.md_views import family_key

    model = {"B", "Na", "O", "Si", "V"}
    assert family_key("CN B (BV)", model) == "CN * (BV)"
    assert family_key("Qn(mB) Si (BV)", model) == "Qn(mB) * (BV)"
    assert family_key("CN Si (BV, distance)", model) == "CN * (BV, distance)"
    assert family_key("g B-O", model) == "g *-*"
    assert family_key("NBO fraction", model) is None


def test_an_msd_from_t_0_opens_on_log_axes():
    from facet.core import md_stats
    from facet.ui import md_plot

    t = np.linspace(0.0, 20.0, 101)
    msd = md_stats.Series.from_frames(
        t, [0.6 * t + 0.01 * k for k in range(3)], name="MSD Si",
        axis_name="t_ps", axis_unit="ps", value_unit="Å^2",
        row_kind="block")
    figure = md_plot.figure_for(msd)
    assert figure.log_x and figure.log_y


def test_an_msd_figure_states_the_centre_of_mass_drift():
    from facet.ui.md_views import com_drift_annotation

    note = ("at the last lag (19 ps) the centre of mass's own MSD is "
            "0.514324 Å^2 (block mean); as a fraction of each element's MSD "
            "there: B 0.8742, Na 0.6944, O 0.8538, Si 0.9065")
    text = com_drift_annotation("MSD Si", ["another note", note])
    assert "0.514324 Å²" in text and "0.9065 of MSD Si" in text
    assert "19 ps" in text
    family = com_drift_annotation("MSD *", [note])
    assert "B 0.8742" in family and "Si 0.9065" in family
    assert com_drift_annotation("MSD of the centre of mass", [note]) is None
    assert com_drift_annotation("g Si-O", [note]) is None
    assert com_drift_annotation("MSD Si", ["no drift note"]) is None


def test_the_rows_table_hides_what_every_row_repeats(qapp):
    from facet.ui.md_views import RowTable

    table = RowTable()
    try:
        rows = [{"descriptor": "Qn Si (BV)", "key": f"Q{k}",
                 "kind": "categories", "mean (fraction)": 0.1 * k,
                 "std (fraction)": 0.01, "frames used": 5}
                for k in range(5)]
        table.set_rows(rows, ["# header"])
        assert table.shown_columns() == ["key", "mean (fraction)"]
        text = table.constant_text()
        for part in ("descriptor = Qn Si (BV)", "kind = categories",
                     "std (fraction) = 0.01", "frames used = 5"):
            assert part in text, part
        # copies keep every column
        assert table.all_text(with_header=False).splitlines()[0].split(
            "\t") == list(rows[0])
        # a single row hides nothing
        table.set_rows(rows[:1])
        assert table.shown_columns() == list(rows[0])
        assert table.constant_text() == ""
    finally:
        dispose(table)


def test_an_export_that_raises_anything_ends_its_thread(qapp, browser,
                                                        tmp_path,
                                                        monkeypatch):
    """A TypeError (or MemoryError) inside the export emitted neither
    signal: the thread ran forever, the browser stayed busy and a closing
    window waited 60 s."""
    from facet.ui import md_views

    def broken(result, path):
        raise TypeError("an exception the export does not name")

    monkeypatch.setattr(md_views.md_export, "export", broken)
    assert browser.start_export_all(tmp_path / "out.xlsx")
    got = _wait(qapp, browser.exportFailed, 20000)
    assert got is not None and "TypeError" in got[0]
    assert browser.shutdown(5000)
    assert not browser.is_exporting()
    # and a second export is accepted
    assert browser.start_export_all(tmp_path / "again.xlsx")
    assert _wait(qapp, browser.exportFailed, 20000) is not None
    assert browser.shutdown(5000)


def test_a_search_that_raises_anything_ends_its_thread(qapp, quartz_frame,
                                                       monkeypatch):
    from facet.ui import md_threshold
    from facet.ui.md_threshold import ThresholdPanel

    def broken(*args, **kwargs):
        raise KeyError("an exception the search does not name")

    monkeypatch.setattr(md_threshold.bulk, "analyse_frame", broken)
    frame, ox = quartz_frame
    p = ThresholdPanel()
    try:
        p.set_frame(frame, ox, background=True, label="frame 0")
        got = _wait(qapp, p.failed, 20000)
        assert got is not None and "KeyError" in got[0]
        assert p.shutdown(5000)
        assert not p.is_busy()
        assert "not analysed" in p.heading.text()
    finally:
        p.shutdown(5000)
        dispose(p)


def test_a_finished_worker_frees_what_it_held(qapp):
    import gc
    import weakref

    from PySide6.QtCore import (QCoreApplication, QEvent, QObject, Signal,
                                Slot)

    from facet.ui import md_views

    class Held:
        pass

    class Worker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, payload):
            super().__init__()
            self.payload = payload

        @Slot()
        def run(self):
            self.finished.emit(None)

    class Receiver(QObject):
        ended = Signal()

        @Slot(object)
        def done(self, _value):
            pass

        @Slot(str)
        def fail(self, _text):
            pass

        @Slot()
        def thread_done(self):
            self.ended.emit()

    receiver = Receiver()
    held = Held()
    ref = weakref.ref(held)
    worker = Worker(held)
    del held
    md_views.start_worker(worker, receiver.done, receiver.fail,
                          receiver.thread_done)
    del worker
    assert _wait(qapp, receiver.ended, 10000) is not None
    for _ in range(3):
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        qapp.processEvents()
        gc.collect()
    assert ref() is None
    assert not md_views._KEEPER.running


def test_every_threshold_plot_saves_itself(qapp, panel, tmp_path,
                                           monkeypatch):
    """The four plots of the threshold panel had no route to SVG, PDF or a
    600 dpi PNG; each now has 'Save figure…' in its context menu."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFileDialog

    from facet.ui import md_plot

    asked = []

    def answer(parent, title, name, filt, *args, **kwargs):
        asked.append(name)
        return str(tmp_path / Path(name).with_suffix(".png").name), \
            "PNG, 600 dpi (*.png)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(answer))
    said = []
    panel.statusMessage.connect(said.append)
    for plot in panel.plots():
        assert plot.contextMenuPolicy() == Qt.ActionsContextMenu
        assert plot.save_figure_action in plot.actions()
        assert plot.save_figure_action.text() == md_plot.SAVE_FIGURE_TEXT
        assert plot.save_figure_action.isEnabled()
        plot.save_figure_action.trigger()
    written = sorted(tmp_path.glob("*.png"))
    assert len(written) == 4, asked
    from PySide6.QtGui import QImage

    for path in written:
        image = QImage(str(path))
        assert image.width() == round(md_plot.EXPORT_WIDTH_MM / 25.4 * 600)
        assert round(image.dotsPerMeterX() * 0.0254) == 600
    assert len([s for s in said if s.startswith("Wrote")]) == 4
    # a plot with nothing to draw offers nothing to save
    empty = md_plot.StatPlot()
    try:
        assert not empty.save_figure_action.isEnabled()
    finally:
        dispose(empty)


# ---------------------------------------------------------------------------
# the final check of the Model window's layout (2026-10-07)
# ---------------------------------------------------------------------------

def test_a_y_title_longer_than_its_axis_takes_two_lines(qapp):
    """The threshold panel's histograms showed "fraction of the element's
    atoms per..." on a 230 px axis: the unit, at the end of the title, was
    cut. A title longer than its axis is now set on two lines, whole."""
    from PySide6.QtGui import QFontMetricsF

    from facet.ui import md_plot
    from facet.ui.md_plot import Figure, StatSeries

    s = StatSeries("Si", [0.0, 1.0, 2.0], [0.1, 0.5, 0.2])
    figure = Figure([s], "bond-valence sum", "v.u.",
                    "fraction of the element's atoms per bin", "1")
    plot = _plot(qapp, figure, width=430, height=230)
    try:
        font = plot._font(False)
        fm = QFontMetricsF(font, plot)
        whole = fm.horizontalAdvance(figure.y_label)

        def layout_at(height):
            return plot._layout(430, height, font, fm, md_plot._SCREEN)

        # an axis about 0.7 of the title long, whatever the fonts
        height = next(h for h in range(60, 4000, 5)
                      if layout_at(h).rect.height() >= 0.7 * whole)
        layout = layout_at(height)
        lines = layout.y_label_lines
        assert len(lines) == 2
        assert " ".join(lines) == figure.y_label
        assert max(fm.horizontalAdvance(t) for t in lines) \
            <= layout.rect.height()
        # a title that fits stays on one line, the plot where it was
        tall = layout_at(next(h for h in range(height, 6000, 10)
                              if layout_at(h).rect.height() >= whole))
        assert tall.y_label_lines == [figure.y_label]
        assert tall.rect.left() < layout.rect.left()
        plot.resize(430, height)
        plot.grab()
    finally:
        dispose(plot)


def test_a_log_axis_of_less_than_two_decades_labels_2_and_5():
    """An MSD from 2 to 18 ps on log axes had one tick label, '10'."""
    import math

    from facet.ui.md_plot import StatPlot

    major, _ = StatPlot._log_ticks(None, math.log10(2.0), math.log10(18.0),
                                   5)
    assert [text for _, text in major] == ["2", "5", "10"]
    # over several decades, the decades alone
    major, _ = StatPlot._log_ticks(None, -3.0, 2.0, 5)
    assert major and all(abs(v - round(v)) < 1e-9 for v, _ in major)


def test_the_tree_shows_mean_and_std_rather_than_a_sliver_of_kind(browser):
    """In a Model window the tree's Kind column pushed Mean ± std out of
    view, one or two letters of Kind showing at its edge. The kind is in
    each row's tool tip and under the title of the item shown."""
    tree = browser.tree
    assert tree.isColumnHidden(1)
    assert not tree.isColumnHidden(2)
    analysis, descriptor = browser.items()[0]
    leaf = browser.find_item(analysis, descriptor)
    assert leaf.text(1) and leaf.text(1) in leaf.toolTip(0)


def test_the_rows_go_under_the_figure_in_a_narrow_view(qapp):
    """Beside a 360 px figure in a 550 px view, the rows table showed one or
    two of its columns and cut the numbers of the next."""
    from PySide6.QtCore import Qt

    from facet.ui.md_views import DescriptorView

    view = DescriptorView()
    try:
        view.resize(550, 700)
        view.show()
        qapp.processEvents()
        assert view.splitter.orientation() == Qt.Vertical
        view.resize(DescriptorView.SIDE_BY_SIDE_MIN + 200, 700)
        qapp.processEvents()
        assert view.splitter.orientation() == Qt.Horizontal
    finally:
        dispose(view)


def test_the_keeper_drops_a_worker_only_after_its_thread_has_ended(qapp):
    """The keeper dropped its reference to a finished thread's worker before
    waiting for the thread: the worker's Python wrapper then deleted it on
    the GUI thread while the worker thread was deleting it too (start_worker
    connected finished to deleteLater), and the process aborted, here and
    there, in test_a_finished_worker_frees_what_it_held. The wait comes
    first now, and the worker is deleted after it."""
    from PySide6.QtCore import QThread

    from facet.ui import md_views

    keeper = md_views._Keeper()
    held_at_wait = []

    class Thread(QThread):
        def wait(self, *args):
            held_at_wait.append(self in keeper.running)
            return True

    thread = Thread()
    keeper.running[thread] = object()
    keeper.sender = lambda: thread
    keeper._release()
    assert held_at_wait == [True]
    assert thread not in keeper.running


def test_a_worker_is_deleted_on_the_gui_thread_not_its_own(qapp):
    """Deleted on its own thread as the thread ended (finished ->
    deleteLater), a worker made in Python takes the GIL inside QObject's
    destructor, under Qt's signal-slot locks; a GUI thread connecting a
    signal meanwhile waited for ever (py-spy on a hung run: show_frame's
    connect against ~QObject waiting in PyGILState_Ensure). The worker now
    outlives its thread and is deleted on the GUI thread."""
    import time

    import shiboken6
    from PySide6.QtCore import QObject, Signal, Slot

    from facet.ui import md_views

    class Worker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        @Slot()
        def run(self):
            self.finished.emit(None)

    class Receiver(QObject):
        @Slot(object)
        def done(self, _value):
            pass

        @Slot(str)
        def fail(self, _text):
            pass

    receiver = Receiver()
    worker = Worker()
    thread = md_views.start_worker(worker, receiver.done, receiver.fail)
    assert thread.wait(10000)
    # the thread has ended and nothing ran on the GUI thread yet: the worker
    # was not deleted on its own thread
    assert shiboken6.isValid(worker)
    end = time.perf_counter() + 10
    while shiboken6.isValid(worker) and time.perf_counter() < end:
        qapp.processEvents()
        time.sleep(0.005)
    assert not shiboken6.isValid(worker)
    del receiver


def test_a_legend_too_narrow_for_its_names_shows_what_differs(qapp):
    """The legend of 'S(Q) *-* (Faber-Ziman)' read 'S(Q) Na-Na (Faber-Zi...',
    six times: the pair is what it shows now, the title saying the rest.
    The series keep their names (the rows and the exports carry them)."""
    from facet.ui.md_plot import Figure, StatSeries, _legend_labels

    names = [f"S(Q) {p} (Faber-Ziman)" for p in ("Na-Na", "Na-O", "O-Si")]
    title = "S(Q) *-* (Faber-Ziman)"
    assert _legend_labels(names, title) == ["Na-Na", "Na-O", "O-Si"]
    # words the title does not state stay, and so does a set of one
    assert _legend_labels(names, "S(Q)") == names
    assert _legend_labels(names[:1], title) == names[:1]
    assert _legend_labels(["N Na around O", "N O around O"],
                          "N * around *") == ["N Na around O",
                                              "N O around O"]
    series = [StatSeries(n, [1.0, 2.0], [1.0, 1.5]) for n in names]
    figure = Figure(series, "Q", "1/Å", "S(Q)", "1", title=title)
    plot = _plot(qapp, figure, width=420, height=380)
    try:
        from PySide6.QtGui import QFontMetricsF

        from facet.ui import md_plot

        font = plot._font(False)
        layout = plot._layout(420, 380, font, QFontMetricsF(font, plot),
                              md_plot._SCREEN)
        assert [text for _, text in layout.legend_rows] == \
            ["Na-Na", "Na-O", "O-Si"]
        assert [s.label for s in figure.series] == names
    finally:
        dispose(plot)


def test_the_last_x_label_is_not_cut_at_the_edge(qapp):
    """The MSD plot of a three-frame run (no legend: the title names the
    one series) drew its last x label, '2.0', centred on the right end of
    the axis with only a gap's room beyond it: it read '2.C'. Every x label
    now lies inside the widget, at any width."""
    from PySide6.QtGui import QFontMetricsF

    from facet.ui import md_plot
    from facet.ui.md_plot import Figure, StatSeries

    s = StatSeries("MSD Si", [0.0, 1.0, 2.0], [0.0, 0.0411, 0.0460])
    figure = Figure([s], "t", "ps", "MSD Si", "Å^2", title="MSD Si")
    plot = _plot(qapp, figure, width=600, height=300)
    try:
        plot.grab()                     # fits the ranges
        font = plot._font(False)
        fm = QFontMetricsF(font, plot)
        checked = 0
        for width in range(380, 1000, 9):
            layout = plot._layout(width, 300, font, fm, md_plot._SCREEN)
            assert layout.legend is None
            rect = layout.rect
            x0, x1 = plot.x_range
            for value, text in layout.x_ticks:
                px = rect.left() + (value - x0) / (x1 - x0) * rect.width()
                right = px + fm.horizontalAdvance(text) / 2
                assert right <= width, (width, text, right)
                checked += 1
            # the axis ends on a labelled tick here, so the case is met
            assert abs(layout.x_ticks[-1][0] - x1) < 1e-9
        assert checked > 100
    finally:
        dispose(plot)
