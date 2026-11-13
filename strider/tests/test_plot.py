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
    for start in (
        "(a) 11 spectra",
        "(b) Best-matching references",
        "(c) Brightness in four bands",
        "(d) Every class",
        "(e) Class",
        "Redshift (68% shaded)",
    ):
        assert any(title.startswith(start) for title in titles), start
    assert figure._suptitle.get_text() == "sn_ia"
    spectra, bands = (next(ax for ax in figure.axes if ax.get_title(loc="left").startswith(t)) for t in ("(a)", "(c)"))
    assert len(spectra.patches) == 4  # the four bands, shaded
    assert [text.get_text()[:11] for text in bands.texts] == ["SN Ia at z "] and len(bands.lines) >= 4
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
