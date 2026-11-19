"""The figure of a result, drawn with the synthetic package, and the figure as the spectra arrive, drawn with the
15-class fixture package."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.colors import to_rgba  # noqa: E402
import strider  # noqa: E402
from strider import plot  # noqa: E402
from strider.model import _steps  # noqa: E402
from strider.tests.synthetic_package import ARCHIVE, CLASSES  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "strider-15class-fixture-1.0.tar.gz"


@pytest.fixture(scope="module")
def model():
    return strider.load_model(ARCHIVE)


@pytest.fixture(scope="module")
def fixture():
    return strider.load_model(FIXTURE)


def columns(figure):
    """Each column's title, S/N text and P(Ia) text, and the class rows of its map."""
    spectra = [ax for ax in figure.axes if ax.get_title().endswith(("spectrum", "spectra"))]
    maps = [ax for ax in figure.axes if ax.get_ylim()[0] > ax.get_ylim()[1]]
    texts = [[t.get_text() for t in ax.texts] for ax in figure.axes if ax.texts]
    labels = [label.get_text() for label in maps[0].get_yticklabels()]
    return [ax.get_title() for ax in spectra], texts[::2], texts[1::2], labels, len(maps)


def test_the_figure_has_its_five_panels(model, tmp_path):
    result = model.classify(strider.load_example())
    figure = result.plot(tmp_path / "result.png", name="sn_ia")
    titles = [ax.get_title(loc="left") for ax in figure.axes]
    for title in ("(a) Spectra", "(b) Best match", "(c) Brightness", "(e) Class"):
        assert title in titles, title
    assert "Redshift" in titles and "(d) Every class at every redshift, relative to the most probable" in titles
    heading = [text.get_text() for text in figure.texts]
    label, z = result.best_match
    best, top = plot._name(label), result._ranked()[0]
    assert heading[:2] == ["sn_ia", result.model]
    tiles = heading[2:14]
    assert tiles[0::3] == ["Class", "Redshift", "Match", "Data"]
    assert tiles[1::3] == [
        f"{plot._name(top[0])} {round(100 * top[1])}%",
        result.report().splitlines()[2].split()[1],  # the report's redshift
        result.match_strength.capitalize(),
        "11 visits",
    ]
    assert tiles[8] == f"best {best}, next {plot._name(result.next_best_class)}"
    notes = [*result.warnings, *result._report_parts()[3]]
    assert heading[14:] == (["Warnings: " + "; ".join(notes)] if notes else [])
    spectra, match, bands = (
        next(ax for ax in figure.axes if ax.get_title(loc="left").startswith(t)) for t in ("(a)", "(b)", "(c)")
    )
    assert len(spectra.patches) == 4  # the four bands, shaded
    assert [text.get_text() for text in spectra.texts] == ["11 spectra: 16–84%", "Combined"]
    assert [text.get_text() for text in match.texts] == ["Combined", f"{best} model"] and len(match.lines) == 2
    assert [text.get_text() for text in bands.texts][0] == f"{best} at z = {z:.2f}" and len(bands.lines) >= 4
    assert bands.get_xlabel() == "Observer-frame days from peak"
    assert (tmp_path / "result.png").stat().st_size > 10_000


@pytest.mark.parametrize("workers", [1, 2])
def test_a_batch_saves_each_figure_and_keeps_none(model, tmp_path, workers):
    objects = strider.load_example(batch=True)
    objects["a/b c"] = objects["sn_ia"]
    results = model.classify_many(objects, plots=tmp_path / "figures", workers=workers)
    altered = "a_b_c-" + hashlib.sha256(b"a/b c").hexdigest()[:8] + ".png"  # 'a_b_c' itself stays free
    assert sorted(path.name for path in (tmp_path / "figures").iterdir()) == sorted(
        [altered, *(f"{key}.png" for key in objects if key != "a/b c")]
    )
    with pytest.raises(ValueError, match="this result has no figure"):
        results["sn_ia"].plot()


def test_a_figure_that_cannot_be_saved_warns_and_the_batch_goes_on(model, tmp_path, monkeypatch):
    from strider import model as module

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(module, "_save_figure", broken)
    with pytest.warns(UserWarning, match="the figure could not be saved"):
        results = model.classify_many(
            strider.load_example(batch=True), tmp_path / "results.parquet", plots=tmp_path / "figures"
        )
    assert all(not isinstance(result, str) for result in results.values())
    assert (tmp_path / "results.parquet").stat().st_size > 0


def redshift_panel(result):
    """The lines of the default figure's redshift panel, by their legend label."""
    ax = next(ax for ax in result.plot().axes if ax.get_title(loc="left").startswith("Redshift"))
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    return dict(zip(labels, ax.get_lines()))


def test_the_redshift_panel_splits_sn_ia_from_the_rest_and_changes_no_number(model):
    result = model.classify(strider.load_example())
    before = json.dumps(result.to_dict())
    lines = redshift_panel(result)
    assert json.dumps(result.to_dict()) == before
    assert list(lines) == ["All classes", "SN Ia", "Other"]
    shown = np.isin(result.redshift_grid, lines["All classes"].get_xdata())
    total = result.redshift_posterior[shown] / result.redshift_cell_widths[shown]
    np.testing.assert_allclose(lines["All classes"].get_ydata(), total / total.max(), rtol=1e-12)
    parts = lines["SN Ia"].get_ydata() + lines["Other"].get_ydata()
    np.testing.assert_allclose(parts, total / total.max(), rtol=1e-12, atol=1e-15)


def test_without_sn_ia_the_redshift_panel_shows_other_as_the_total(model):
    lines = redshift_panel(model.classify(strider.load_example(), classes=["IIP", "Ic"]))
    assert list(lines) == ["All classes", "Other"]
    np.testing.assert_allclose(lines["Other"].get_ydata(), lines["All classes"].get_ydata(), rtol=1e-12)


def by_date(transient, k):
    """The transient's first ``k`` visits by MJD."""
    return replace(transient, visits=sorted(transient.visits, key=lambda visit: visit.mjd)[:k])


def panel(figure, start):
    return next(ax for ax in figure.axes if ax.get_title(loc="left").startswith(start))


def red_edge(model, transient):
    """The observed wavelength, in microns, where the spectra panel ends."""
    result = model.classify(transient)
    return panel(result.plot(), "(a)").get_xlim()[1] * (1.0 + result.redshift)


def test_pixels_of_huge_error_are_left_out_of_the_spectra(model):
    example = strider.load_example()
    huge = [np.where(visit.wavelength > 17000, 1e7, 1.0) * visit.error for visit in example.visits]
    visits = [replace(visit, error=error) for visit, error in zip(example.visits, huge)]
    assert red_edge(model, example) > 1.75
    assert red_edge(model, replace(example, visits=visits)) < 1.70


def test_brightness_points_below_s_n_2_are_open_and_the_panel_says_so(model):
    example = strider.load_example()

    def drawn(factor):
        """The note under (c)'s label, and each point's S/N and whether it is drawn open."""
        visits = [replace(visit, error=visit.error * factor) for visit in example.visits]
        bands = panel(model.classify(replace(example, visits=visits)).plot(), "(c)")
        points = []
        for container in bands.containers:
            line, _, (bars,) = container.lines
            open_ = to_rgba(line.get_markerfacecolor()) == to_rgba("white")
            for y, segment in zip(line.get_ydata(), bars.get_segments()):
                points.append((y / ((segment[1, 1] - segment[0, 1]) / 2), open_))
        return [text.get_text() for text in bands.texts][1:], points

    note, points = drawn(10.0)  # some points measured, most noise
    assert {open_ for _, open_ in points} == {True, False} and note == ["Open circles: S/N < 2"]
    assert all(open_ == (ratio < 2.0) for ratio, open_ in points)
    note, points = drawn(1e3)  # every point noise: all open
    assert note == ["Open circles: S/N < 2"] and all(open_ for _, open_ in points)


def test_with_one_visit_the_figure_shows_the_spectrum_and_says_why_it_has_no_fit(model):
    figure = model.classify(by_date(strider.load_example(), 1)).plot()
    assert [text.get_text() for text in panel(figure, "(a)").texts] == ["Spectrum"]  # no spread, nothing combined
    bands = panel(figure, "(c)")
    assert [text.get_text() for text in bands.texts] == ["Too few visits to fit"]
    assert bands.get_xlabel() == "Observer-frame days"


def same(result, expected):
    """Whether two results agree in every field but ``visits_used``."""
    first, second = result.to_dict(), expected.to_dict()
    del first["visits_used"], second["visits_used"]
    return json.dumps(first) == json.dumps(second)


def test_the_accumulation_figure_has_a_column_for_each_step(fixture, tmp_path):
    figure = fixture.plot_accumulation(
        strider.load_example(), path=tmp_path / "arrival.png", name="sn_ia", true_redshift=0.5
    )
    titles, signal, probability, labels, maps = columns(figure)
    assert titles == ["1 spectrum", "2 spectra", "3 spectra", "5 spectra", "10 spectra", "11 spectra"]
    assert labels == ["SN Ia", "Other"] and maps == 6
    assert all(text[0].startswith("S/N") for text in signal)
    assert all(text[0].startswith(r"$P(\mathrm{Ia}) = ") for text in probability)
    assert figure._suptitle.get_text() == "sn_ia"
    assert (tmp_path / "arrival.png").stat().st_size > 10_000


def test_each_column_is_classify_of_the_first_spectra_by_date(fixture, monkeypatch):
    example = strider.load_example()
    order = np.random.default_rng(1).permutation(len(example.visits))
    visits = [example.visits[i] for i in order]
    arrays = [[getattr(visit, name) for visit in visits] for name in ("wavelength", "flux", "error", "mjd")]
    drawn = {}
    monkeypatch.setattr(plot, "accumulation", lambda results, *args: drawn.update(results) or "figure")
    assert fixture.plot_accumulation(*arrays, steps=[1, 2, 4, 11]) == "figure"
    assert list(drawn) == [1, 2, 4, 11]
    for k, result in drawn.items():
        assert same(result, fixture.classify(by_date(example, k)))
        assert result.visits_used == np.argsort(arrays[3])[:k].tolist()


def test_a_spectrum_left_out_is_skipped(fixture):
    example = strider.load_example()
    earliest = example.visits[0]
    short = replace(
        earliest, wavelength=earliest.wavelength[:100], flux=earliest.flux[:100], error=earliest.error[:100]
    )
    transient = replace(example, visits=[short, *example.visits[1:]])
    results = fixture.classify_first(transient)
    assert list(results) == [1, 2, 3, 5, 10]
    assert results[1].visits_used == [1] and 0 in results[10].dropped_visits
    with pytest.raises(ValueError, match="steps go up to 11 spectra, but this transient has 10 that can be used"):
        fixture.classify_first(transient, steps=[2, 11])


@pytest.mark.parametrize(
    "priors",
    [
        {"host_redshift": 0.5, "host_redshift_error": 0.05},
        {"class_prior": dict.fromkeys(CLASSES, 1.0) | {"Ia": 5.0}},
        {"redshift_range": (0.3, 1.5)},
        {"classes": ["Ia", "IIP", "Ic"]},
    ],
)
def test_classify_first_passes_the_priors_on(fixture, priors):
    example = strider.load_example()
    result = fixture.classify_first(example, steps=[3], **priors)[3]
    assert same(result, fixture.classify(by_date(example, 3), **priors))
    assert not same(result, fixture.classify(by_date(example, 3)))


def test_classify_first_uses_the_host_a_transient_carries(fixture):
    example = strider.load_example()
    hosted = replace(example, host_redshift=0.5, host_redshift_error=0.05)
    result = fixture.classify_first(hosted, steps=[3])[3]
    assert same(result, fixture.classify(by_date(hosted, 3)))
    assert not same(result, fixture.classify(by_date(example, 3)))


@pytest.mark.parametrize(
    ("count", "steps"),
    [(1, [1]), (2, [1, 2]), (6, [1, 2, 3, 5, 6]), (20, [1, 2, 3, 5, 10, 20]), (21, [1, 3, 5, 10, 20, 21])],
)
def test_the_default_steps(count, steps):
    assert _steps(None, count) == steps


def test_a_column_measured_only_in_the_tapered_ends_is_drawn(fixture):
    example = strider.load_example()
    earliest = example.visits[0]
    transient = replace(example, visits=[replace(earliest, mask=earliest.wavelength > 7620.0), *example.visits[1:]])
    measured = fixture.classify_first(transient, steps=[1])[1]._source[1].measured[0]
    assert measured.any() and (fixture.network.matching_weight.numpy()[measured] <= 0.5).all()
    figure = fixture.plot_accumulation(transient, steps=[1, 2])
    assert columns(figure)[0] == ["1 spectrum", "2 spectra"]


def test_every_row_shows_only_the_classes_allowed(fixture):
    example = strider.load_example()
    figure = fixture.plot_accumulation(example, rows="all", classes=["Ia", "IIP", "Ic"], steps=[2, 3])
    _, _, probability, labels, _ = columns(figure)
    assert labels == ["SN Ia", "IIP", "Ic"]
    for k, text in zip((2, 3), probability):
        assert f"= {fixture.classify(by_date(example, k), classes=['Ia', 'IIP', 'Ic']).p_ia:.2f}$" in text[0]
    _, _, _, labels, _ = columns(fixture.plot_accumulation(example, rows="all", steps=[2]))
    assert labels == ["SN Ia", *fixture.classes[1:]]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"steps": [2, 12]}, "steps go up to 12 spectra, but this transient has 11"),
        ({"steps": [3, 2]}, "in increasing order"),
        ({"steps": [0, 2]}, "at least 1"),
        ({"steps": [True]}, "whole and at least 1"),
        ({"steps": 5}, "must be a sequence of numbers of spectra"),
        ({"rows": "some"}, 'rows must be "binary" or "all"'),
        ({"true_redshift": 7.0}, "true_redshift 7.0 must be a number within the model's redshifts"),
    ],
)
def test_the_accumulation_figure_refuses_what_it_cannot_draw(fixture, options, message):
    with pytest.raises(ValueError, match=message):
        fixture.plot_accumulation(strider.load_example(), **options)


def test_the_main_redshift_ring_is_bolder_than_the_second(model):
    result = model.classify(strider.load_example())
    two = replace(result, redshift_peaks=[(0.5, 0.6), (2.0, 0.3)])
    two._source = result._source
    ax = panel(two.plot(), "(d)")
    rings = {line.get_label(): line for line in ax.get_lines()}
    main, second = rings["main redshift"], rings["second redshift"]
    assert main.get_markersize() > second.get_markersize() and main.get_markeredgewidth() > second.get_markeredgewidth()
    assert main.get_alpha() == 1.0 and second.get_alpha() < 1.0 and main.get_path_effects()  # seen on any cell


def test_one_very_noisy_spectrum_does_not_set_the_spread():
    from types import SimpleNamespace

    flux, log_error = np.ones((3, 100)), np.zeros((3, 100))
    flux[0, 40:45], flux[0, 45:50] = -1e6, 1e6  # far below and above zero, summing to nothing...
    log_error[0, 40:50] = np.log(1e7)  # ...with errors to match
    spectra = SimpleNamespace(flux=flux, measured=np.ones((3, 100), bool), log_error=log_error)
    low, high = plot._spread(spectra, np.ones(100), np.ones(100, bool), 1.0)
    assert np.nanmin(low) > 0.5 and np.nanmax(high) < 1.5


def test_many_warnings_wrap_under_the_box(model):
    result = model.classify(strider.load_example())
    many = replace(result, warnings=[f"a long warning about something odd, number {n}" for n in range(6)])
    many._source = result._source
    lines = [text.get_text() for text in many.plot().texts if text.get_text().startswith("Warnings: ")][0]
    assert len(lines.splitlines()) > 1 and max(map(len, lines.splitlines())) <= 105


def test_a_few_deep_noise_spikes_do_not_set_the_spectrum_panels_height():
    import matplotlib.pyplot as plt

    line = np.full(200, 0.6)
    line[:3] = -5.0  # 1.5% of the pixels
    figure, ax = plt.subplots()
    plot._fill_height(ax, line)
    assert ax.get_ylim()[0] == pytest.approx(-0.08)
    plot._fill_height(ax, line, np.array([-0.5, 1.0]))  # a model line is shown whole
    assert ax.get_ylim()[0] == pytest.approx(-0.58)
    plt.close(figure)
