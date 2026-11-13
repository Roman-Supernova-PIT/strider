"""Classifying with the synthetic package: one transient, a batch, and the batch tables."""

import json
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")

import strider  # noqa: E402
from strider import posterior  # noqa: E402
from strider.io import Transient, Visit, read_spectra, results_table, write_spectra  # noqa: E402
from strider.preprocess import measured_error_scale  # noqa: E402
from strider.tests.synthetic_package import ARCHIVE, package_files, write_directory  # noqa: E402

EXAMPLE = files("strider") / "data" / "example.parquet"


@pytest.fixture(scope="module")
def model():
    return strider.load_model(ARCHIVE)


def lists(transient):
    return [[getattr(visit, name) for visit in transient.visits] for name in ("wavelength", "flux", "error", "mjd")]


def test_one_transient(model):
    data = lists(read_spectra(EXAMPLE)["sn_ia"])
    result = model.classify(*data)
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0)
    assert sum(result.equal_prior_probabilities.values()) == pytest.approx(1.0)
    assert result.p_ia == result.class_probabilities["Ia"]
    assert result.redshift_posterior.sum() == pytest.approx(1.0)
    assert result.redshift_cell_widths.sum() == pytest.approx(result.redshift_grid[-1] - result.redshift_grid[0])
    low95, high95 = result.redshift_95
    assert low95 <= result.redshift_68[0] <= result.redshift_68[1] <= high95
    assert (result.redshift, result.redshift_probability) == result.redshift_peaks[0]
    assert [p for _, p in result.redshift_peaks] == sorted((p for _, p in result.redshift_peaks), reverse=True)
    assert result.flags["several_redshift_peaks"] == (len(result.redshift_peaks) > 1)
    assert result.class_redshifts["Ia"] == result.redshift_given("Ia").redshift
    assert result.log_evidence.shape == result.class_redshift_posterior.shape
    assert result.days_observed > 0 and np.isfinite(result.signal_to_noise)
    assert 0.0 < result.source_probability < 1.0
    assert result.flags["weak_source"] == (result.source_probability < model.weak_source_below)
    assert set(result.flags) == {
        "several_redshift_peaks",
        "weak_source",
        "untrained_subtypes",
        "few_classes_trained",
        "brightness_failed",
    }
    assert set(result.trained_classes) <= set(model.classes)
    assert result.visits_used == list(range(11)) and result.dropped_visits == {}
    json.dumps(result.to_dict())

    host = float(model.redshift_grid[8])
    hosted = model.classify(*data, host_redshift=host, host_redshift_error=0.01)
    assert hosted.redshift_posterior[8] > 0.9  # 5% is left for the host being wrong
    with pytest.raises(ValueError, match="together"):
        model.classify(*data, host_redshift=host)
    with pytest.raises(ValueError, match="must be numbers"):
        model.classify(*data, host_redshift="0.5", host_redshift_error=0.01)
    with pytest.raises(ValueError, match="more than 5 errors outside the model's redshift range"):
        model.classify(*data, host_redshift=3.06, host_redshift_error=0.01)
    trusted = model.classify(*data, host_redshift=3.04, host_redshift_error=0.01, host_redshift_outlier=0.0)
    assert trusted.redshift_posterior[-1] > 0.9
    with pytest.raises(ValueError, match="from 0 up to 1"):
        model.classify(*data, host_redshift=0.5, host_redshift_error=0.01, host_redshift_outlier=1.0)


def test_a_batch_survives_a_bad_object_and_its_tables_round_trip(model, tmp_path):
    objects = read_spectra(EXAMPLE)
    host = float(model.redshift_grid[8])
    objects["hosted"] = Transient(objects["sn_ia"].visits, host, 0.01)
    objects["short"] = Transient([Visit(np.geomspace(8000.0, 9000.0, 50), np.ones(50), np.ones(50), 60000.0)])
    write_spectra(objects, tmp_path / "spectra.parquet")
    found = read_spectra(tmp_path / "spectra.parquet")
    assert list(found) == list(objects)
    assert (found["hosted"].host_redshift, found["hosted"].host_redshift_error) == (host, 0.01)
    assert found["sn_ia"].host_redshift is None and len(found["sn_ia"].visits) == 11

    results = model.classify_many(found, tmp_path / "results.parquet")
    assert results["short"].startswith("ValueError: no usable visit")
    alone = model.classify(*lists(objects["sn_ia"]))
    assert results["sn_ia"].class_probabilities == alone.class_probabilities
    assert results["hosted"].redshift_posterior[8] > 0.9
    table = pq.read_table(tmp_path / "results.parquet")
    rows = {row["object_id"]: row for row in table.to_pylist()}
    assert rows["short"]["failure"].startswith("ValueError") and rows["short"]["p_ia"] is None
    assert rows["sn_ia"]["p_ia"] == alone.p_ia and rows["sn_ia"]["failure"] is None
    assert rows["hosted"]["fingerprint"] == model.fingerprint
    assert {
        "redshift_95_lower",
        "untrained_share",
        "untrained_subtypes",
        "equal_prior_Ia",
        "few_classes_trained",
        "brightness_failed",
    } <= set(table.column_names)
    assert len(json.loads(table.schema.metadata[b"redshift_grid"])) == len(model.redshift_grid)


def test_a_survey_prior_without_some_class_refuses_the_package(tmp_path):
    def without_kn(priors):
        priors["survey"]["class_fraction"]["KN"] = 0.0
        return priors

    path = write_directory(tmp_path / "package", package_files(), **{"priors.json": without_kn})
    with pytest.raises(ValueError, match=f"model package {path}: priors.json gives KN no share"):
        strider.load_model(path)


def test_mps_is_refused():
    with pytest.raises(ValueError, match="MPS has no float64"):
        strider.load_model(ARCHIVE, device="mps")


def test_the_readme_quickstart_runs_as_written(model, tmp_path, monkeypatch, capsys):
    import re
    from pathlib import Path

    from strider import registry
    from strider.cli import main

    readme = Path(__file__).parents[2] / "README.rst"
    if not readme.exists():
        pytest.skip("README.rst is not installed with the package")
    pytest.importorskip("matplotlib").use("Agg")
    lines = readme.read_text().splitlines()

    def block(start):
        found = []
        for line in lines[lines.index(start) + 2 :]:
            if line and not line.startswith("    "):
                return found
            found.append(line[4:])

    def labels(card):
        """Each line's label, once every line has the card's shape."""
        shapes = {
            "Class": r"[\w-]+ (\d+|<1)%( · [\w-]+ (\d+|<1)%){0,2}",
            "Redshift": r"\d\.\d+ \((\d+|<1)% within 0\.1\) · 68% range \d\.\d+–\d\.\d+",
            "Match": r"(high|medium|low) strength( · .+)?",
            "Data": r"\d+ visits?( over \d+ days| on one night)?( · S/N \d+\.\d)?( · errors × \d+\.\d\d)?",
            "Checks": r".+",
        }
        assert len(card[0]) == 79 and card[0].startswith("STRIDER · transient ")
        for line in card[1:]:
            assert re.fullmatch(shapes[line[:10].rstrip()], line[10:]), line
        return [line[:10].rstrip() for line in card[1:]]

    # the test model stands in for the released one, which load_model() downloads
    fetched = []
    monkeypatch.setattr(registry, "fetch", lambda name=None: fetched.append(name) or ARCHIVE)
    monkeypatch.chdir(tmp_path)
    exec("\n".join(block(".. code-block:: python")), {})
    printed = capsys.readouterr().out.splitlines()
    assert printed == model.classify(strider.load_example()).card().splitlines()
    assert (tmp_path / "example.png").stat().st_size > 10_000
    shown = [line for line in block(".. code-block:: text") if line]
    released = Path(registry.MODELS[registry.DEFAULT_MODEL]["url"]).name.removesuffix(".tar.gz")
    assert shown[0].endswith(released) and printed[0].endswith(model.name)
    assert labels(shown) == labels(printed)[:4] == ["Class", "Redshift", "Match", "Data"]
    # the command line names the transient after its file
    command = next(line[4:] for line in lines if line.startswith("    strider classify "))
    name = Path(command.split()[-1]).stem
    (tmp_path / f"{name}.csv").write_bytes((files("strider") / "data" / "example.csv").read_bytes())
    main(command.split()[1:])
    assert capsys.readouterr().out.splitlines() == [
        printed[0].replace("transient".ljust(len(name)), name, 1),
        *printed[1:],
    ]
    assert fetched == [None, None]


def test_neither_the_flux_unit_nor_the_order_of_visits_on_one_date_changes_the_result(model):
    wavelength, flux, error, mjd = lists(read_spectra(EXAMPLE)["sn_ia"])
    base = model.classify(wavelength, flux, error, mjd)
    for unit in (3.0, 1e-17):
        scaled = model.classify(wavelength, [f * unit for f in flux], [e * unit for e in error], mjd)
        np.testing.assert_allclose(
            list(scaled.class_probabilities.values()), list(base.class_probabilities.values()), rtol=0, atol=1e-6
        )
        np.testing.assert_allclose(scaled.redshift_posterior, base.redshift_posterior, rtol=0, atol=1e-6)
        assert scaled.flags == base.flags and scaled.visits_used == base.visits_used

    same_date = list(mjd)
    same_date[3] = same_date[4]
    swap = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10]
    first = model.classify(wavelength, flux, error, same_date)
    second = model.classify(*([values[i] for i in swap] for values in (wavelength, flux, error, same_date)))
    assert second.class_probabilities == first.class_probabilities
    assert np.array_equal(second.redshift_posterior, first.redshift_posterior)
    assert [swap[i] for i in second.visits_used] == first.visits_used


def noisier(transient, factor=1.5, seed=7):
    """The transient with white noise added, so that it scatters ``factor`` times its errors."""
    rng = np.random.default_rng(seed)
    extra = np.sqrt(factor**2 - 1.0)
    visits = [replace(v, flux=v.flux + extra * v.error * rng.normal(size=len(v.flux))) for v in transient.visits]
    return replace(transient, visits=visits)


def test_the_error_scale_is_off_by_default_and_reaches_every_path(model):
    objects = read_spectra(EXAMPLE)
    wavelength, flux, error, mjd = data = lists(objects["sn_ia"])
    plain = model.classify(*data)
    same = model.classify(*data, error_scale=1.0)
    assert same.class_probabilities == plain.class_probabilities and plain.error_scale == same.error_scale == 1.0
    assert np.array_equal(same.redshift_posterior, plain.redshift_posterior)

    # a scale multiplies the errors given; "auto" measures it from each transient's own spectra
    doubled = model.classify(wavelength, flux, [2 * part for part in error], mjd)
    scaled = model.classify(*data, error_scale=2.0)
    assert scaled.class_probabilities == doubled.class_probabilities and scaled.error_scale == 2.0
    assert doubled.p_ia != plain.p_ia
    noisy = {key: noisier(transient) for key, transient in objects.items()}
    alone = {key: model.classify(transient, error_scale="auto") for key, transient in noisy.items()}
    for key, result in alone.items():
        assert 1.2 < result.error_scale < 1.8, key
        assert result.p_ia != model.classify(noisy[key]).p_ia
    for batch, workers in ((1, 1), (3, 1), (1, 2)):
        many = model.classify_many(noisy, batch=batch, workers=workers, error_scale="auto")
        for key, result in alone.items():
            assert many[key].p_ia == pytest.approx(result.p_ia, abs=1e-6)
            assert many[key].error_scale == result.error_scale
    transient = noisy["sn_ia"]
    first = model.classify_first(transient, steps=[3, 11], error_scale="auto")
    for k in (3, 11):
        part = model.classify(
            replace(transient, visits=[transient.visits[i] for i in first[k].visits_used]), error_scale="auto"
        )
        assert first[k].class_probabilities == part.class_probabilities and first[k].error_scale == part.error_scale
    shown = f"Data      11 visits over 100 days · S/N {scaled.signal_to_noise:.1f} · errors × 2.00"
    assert shown in scaled.card().splitlines() and "errors" not in plain.card()
    for wrong in (0, True, np.nan, 1e4, 1 + 2j, "2", "AUTO", None):
        with pytest.raises(ValueError, match="error_scale"):
            model.classify(*data, error_scale=wrong)
        with pytest.raises(ValueError, match="error_scale"):
            model.classify_many(objects, error_scale=wrong)


def test_auto_measures_the_error_scale_on_the_visits_kept(model):
    transient = noisier(read_spectra(EXAMPLE)["sn_ia"])
    # a visit covering only the blue end is dropped; it has more pixels than the rest and scatters 3 times its errors
    first = transient.visits[0]
    wavelength = np.linspace(first.wavelength[0], 11000.0, 6000)
    error = np.full(len(wavelength), float(np.median(first.error)))
    blue = Visit(wavelength, 3.0 * error * np.random.default_rng(1).normal(size=len(error)), error, first.mjd + 0.5)
    every = [*transient.visits, blue]
    assert measured_error_scale(*([getattr(v, k) for v in every] for k in ("wavelength", "flux", "error"))) > 2.0
    full = model.classify(replace(transient, visits=every), error_scale="auto")
    assert list(full.dropped_visits) == [11] and 1.2 < full.error_scale < 1.8
    alone = model.classify(transient, error_scale="auto")
    assert full.error_scale == alone.error_scale and full.class_probabilities == alone.class_probabilities
    last = model.classify_first(replace(transient, visits=every), steps=[3, 11], error_scale="auto")[11]
    assert last.error_scale == alone.error_scale and last.class_probabilities == alone.class_probabilities


def test_two_visits_on_one_date_change_smoothly_with_the_error_scale(model):
    # their order, and so the null copies, would otherwise flip as the errors are scaled
    wavelength, flux, error, mjd = lists(read_spectra(EXAMPLE)["sn_ia"])
    mjd[3] = mjd[4]
    p_ia = [model.classify(wavelength, flux, error, mjd, error_scale=scale).p_ia for scale in np.linspace(1.2, 1.25, 6)]
    assert np.abs(np.diff(p_ia, 2)).max() < 5e-4


def test_a_worker_refuses_a_package_whose_files_changed_though_its_fingerprint_did_not(tmp_path, monkeypatch):
    from strider import model as module

    files = package_files()
    first = write_directory(tmp_path / "first", files)
    loaded = strider.load_model(first)

    def recalibrated(calibration):
        return calibration | {"source_probability": calibration["source_probability"] | {"intercept": 0.0}}

    changed = write_directory(tmp_path / "changed", files, **{"calibration.json": recalibrated})
    assert strider.load_model(changed).fingerprint == loaded.fingerprint
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)  # as a worker would, but not here
    monkeypatch.setattr(module, "_worker_model", None)
    module._start_worker(first, loaded.files_sha256, False)
    with pytest.raises(RuntimeError, match="changed after it was loaded"):
        module._start_worker(changed, loaded.files_sha256, False)


def test_workers_give_the_results_of_one_process_in_order(model):
    objects = read_spectra(EXAMPLE)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # as each worker runs
    try:
        alone = model.classify_many(objects)
    finally:
        torch.set_num_threads(threads)
    together = model.classify_many(objects, workers=2)
    assert list(together) == list(objects)
    for key, result in alone.items():
        assert together[key].class_probabilities == result.class_probabilities
        assert np.array_equal(together[key].redshift_posterior, result.redshift_posterior)
    with pytest.raises(ValueError, match="workers"):
        model.classify_many(objects, workers=0)


def test_transients_scored_together_give_the_results_of_one_at_a_time(model, monkeypatch):
    objects = read_spectra(EXAMPLE)  # different numbers of visits; in twos, the last one is alone
    alone, together = model.classify_many(objects), model.classify_many(objects, batch=2)
    assert list(together) == list(objects)
    for key, result in alone.items():
        np.testing.assert_allclose(together[key].log_evidence, result.log_evidence, rtol=0, atol=1e-5)
        assert together[key].redshift == pytest.approx(result.redshift, abs=1e-6)
    with pytest.raises(ValueError, match="batch"):
        model.classify_many(objects, batch=0)
    with pytest.raises(ValueError, match="not both"):
        model.classify_many(objects, workers=2, batch=2)

    sizes, evaluate = [], model._evaluate_many

    def on_the_cpu(prepared):  # on a GPU model; only the default batch reads the device here
        sizes.append(len(prepared))
        model.device = "cpu"
        try:
            return evaluate(prepared)
        finally:
            model.device = "cuda"

    monkeypatch.setattr(model, "_evaluate_many", on_the_cpu)
    monkeypatch.setattr(model, "device", "cuda")
    many = {f"{key}-{n}": transient for n in range(12) for key, transient in objects.items()}
    assert len(model.classify_many(many)) == len(many)
    assert sizes == [8, 8, 8, 8, len(many) - 32]


def test_a_batch_out_of_gpu_memory_is_halved_and_tried_again(model, monkeypatch):
    """On the CPU, with running out of GPU memory faked above two transients at once."""
    import torch

    objects = strider.load_example(batch=True)
    alone = model.classify_many(objects)
    sizes, evaluate = [], model._evaluate_many

    def small_gpu(prepared):
        sizes.append(len(prepared))
        if len(prepared) > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (faked)")
        return evaluate(prepared)

    monkeypatch.setattr(model, "_evaluate_many", small_gpu)
    with pytest.warns(UserWarning, match="out of GPU memory scoring 3 transients together; trying 1"):
        halved = model.classify_many(objects, batch=3)
    assert sizes == [3, 1, 1, 1]
    for key, result in alone.items():
        assert halved[key].redshift == pytest.approx(result.redshift, abs=1e-6)
        assert halved[key].p_ia == pytest.approx(result.p_ia, abs=1e-6)
    monkeypatch.setattr(
        model,
        "_evaluate_many",
        lambda prepared: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("CUDA out of memory (faked)")),
    )
    with pytest.raises(torch.cuda.OutOfMemoryError):
        model._scored([None], 1)


def test_a_batch_out_of_gpu_memory_is_freed_before_it_is_tried_again():
    """The failed attempt's tensors, held by the traceback while it is handled, are gone before the smaller batch."""
    import weakref

    import torch

    attempts, held = [], []

    class Scorer:
        _scored = strider.Model._scored

        def _evaluate_many(self, part):
            held.append(sum(ref() is not None for ref in attempts))
            work = torch.empty(1000)  # stands for the batch's intermediates, in the frame that fails
            attempts.append(weakref.ref(work))
            if len(part) > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (faked)")
            return [{}]

    with pytest.warns(UserWarning, match="trying 2"):
        assert len(Scorer()._scored(list(range(4)), 4)) == 4
    assert held == [0, 0, 0, 0, 0, 0]  # 4 fails, then 2 fails, then four of 1: the halved size is kept


def test_a_transient_the_network_cannot_score_gets_its_reason_and_the_batch_goes_on(model, monkeypatch):
    import torch

    objects = strider.load_example(batch=True)
    alone = model.classify_many(objects)
    evaluate = model._evaluate_many

    def failing(prepared):  # the eleven-visit transient cannot be scored, alone or with others
        if any(len(p.days) == 11 for p in prepared):
            raise torch.linalg.LinAlgError("linalg.eigh: the algorithm failed to converge (faked)")
        return evaluate(prepared)

    monkeypatch.setattr(model, "_evaluate_many", failing)
    together = model.classify_many(objects, batch=3)
    assert list(together) == list(objects)
    assert "linalg.eigh: the algorithm failed to converge" in together["sn_ia"]
    for key in ("masked_and_bad_pixels", "a_masked_block"):
        assert together[key].p_ia == pytest.approx(alone[key].p_ia, abs=1e-6)


def test_on_a_gpu_the_network_runs_without_tf32():
    import torch

    from strider.model import _exact_float32

    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        with _exact_float32("cuda:0"):
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        with _exact_float32("cpu"):
            assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def test_a_transient_missing_a_band_others_measure_scores_as_it_does_alone(model):
    """Scored together, a transient whose red half is masked gets the empty red band of the union: its coadd term
    stays as alone to float32 rounding, and its brightness term, scored on its own, is the same."""
    transient = strider.load_example(batch=True)["sn_ia"]
    red_masked = [visit.wavelength > 12500.0 for visit in transient.visits]
    whole, half = model._prepare(*lists(transient), None), model._prepare(*lists(transient), red_masked)
    alone, together = model._evaluate(half), model._evaluate_many([whole, half])[1]
    assert np.isnan(half.band_flux[:, 2:]).all() and np.isfinite(half.band_flux[:, :2]).all()
    np.testing.assert_array_equal(together["brightness"], alone["brightness"])
    np.testing.assert_allclose(together["coadd_term"], alone["coadd_term"], rtol=0, atol=1e-5)


def test_the_result_fuses_the_spectrum_and_the_brightness_term(model, monkeypatch):
    """The evidence map is a (log q - log pi) + w L, up to a constant, on the cells with training transients and -inf
    elsewhere; where the brightness term fails, L is 0 and the spectrum stands alone, which the result records."""
    import torch

    transient = strider.load_example()
    output = model._evaluate(model._prepare(*lists(transient), None))
    scored = np.isfinite(model.prior["training"])
    assert np.array_equal(np.isfinite(output["brightness"]), scored) and np.ptp(output["brightness"][scored]) > 0.1
    spectrum = np.where(output["support"], output["coadd_term"], posterior.UNSUPPORTED)
    spectrum = posterior.log_posterior(spectrum, model.widths, model.measure) - model.prior["training"]

    def fused(evidence, brightness):
        brightness = np.broadcast_to(brightness, scored.shape)[scored]
        difference = evidence[scored] - model.fusion[0] * spectrum[scored] - model.fusion[1] * brightness
        return np.ptp(difference) < 1e-9 and np.isneginf(evidence[~scored]).all()

    result = model.classify(transient)
    assert fused(result.log_evidence, output["brightness"]) and not result.flags["brightness_failed"]
    monkeypatch.setattr(model.brightness, "score", lambda *_: (_ for _ in ()).throw(torch.linalg.LinAlgError("faked")))
    with pytest.warns(UserWarning, match=r"the brightness term could not be computed \(faked\)"):
        failed = model.classify(transient)
    assert fused(failed.log_evidence, 0.0) and failed.to_dict()["flags"]["brightness_failed"]
    checks = failed.card().splitlines()[-1]
    assert checks.startswith("Checks    ") and checks.endswith("brightness failed, spectra alone")
    assert "brightness failed" not in result.card()
    assert results_table({"one": failed}, model, summary=True).column("brightness_failed").to_pylist() == [True]


def test_a_stop_during_one_file_s_write_writes_no_transient_twice(model, tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    from strider import io

    objects, write = strider.load_example(batch=True), io.EvidenceMapsFile.write

    def stopped(self, results):  # the scheduler's stop lands in the evidence-map write, after the results'
        monkeypatch.setattr(io.EvidenceMapsFile, "write", write)
        raise SystemExit(143)

    monkeypatch.setattr(io.EvidenceMapsFile, "write", stopped)
    with pytest.raises(SystemExit):
        model.classify_many(objects, tmp_path / "out.parquet", evidence_maps=tmp_path / "maps.h5", chunk=3)
    assert pq.read_table(tmp_path / "out.parquet").column("object_id").to_pylist() == list(objects)


def test_a_batch_that_stops_first_writes_the_transients_done(model, tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    objects, each = read_spectra(EXAMPLE), model._each

    def stopped(*args):
        for n, found in enumerate(each(*args)):
            if n == 2:
                raise KeyboardInterrupt
            yield found

    monkeypatch.setattr(model, "_each", stopped)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=5)
    assert pq.read_table(tmp_path / "out.parquet").column("object_id").to_pylist() == list(objects)[:2]


def test_every_batch_gives_one_schema_with_the_grid(model, tmp_path):
    import pyarrow as pa

    objects = read_spectra(EXAMPLE)
    short = Transient([Visit(np.geomspace(8000.0, 9000.0, 50), np.ones(50), np.ones(50), 60000.0)])
    tables = []
    for name, batch in (("mixed", objects | {"short": short}), ("failing", {"short": short}), ("empty", {})):
        model.classify_many(batch, tmp_path / f"{name}.parquet", chunk=2)
        tables.append(pq.read_table(tmp_path / f"{name}.parquet"))
    assert [table.num_rows for table in tables] == [4, 1, 0]
    assert all(table.schema.equals(tables[0].schema, check_metadata=True) for table in tables)
    metadata = tables[2].schema.metadata
    np.testing.assert_allclose(json.loads(metadata[b"redshift_grid"]), model.redshift_grid)
    np.testing.assert_allclose(json.loads(metadata[b"redshift_cell_widths"]), model.widths)
    combined = pa.concat_tables(tables)
    assert combined.num_rows == 5 and combined.schema.field("failure").type == pa.string()
    assert combined.schema.field("dropped_visits").type == pa.list_(pa.string())


def test_a_table_of_spectra_is_checked_as_it_is_read(tmp_path):
    import pyarrow as pa

    rows = pq.read_table(EXAMPLE).to_pylist()
    pq.write_table(
        pa.Table.from_pylist([{k: v for k, v in row.items() if k != "error"} for row in rows]),
        tmp_path / "no_error.parquet",
    )
    with pytest.raises(ValueError, match="is missing the column.s. error. A table of spectra needs one row per visit"):
        read_spectra(tmp_path / "no_error.parquet")
    rows[3]["mjd"] = None
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "no_mjd.parquet")
    with pytest.raises(ValueError, match="a visit of object sn_ia has no finite mjd"):
        read_spectra(tmp_path / "no_mjd.parquet")
    found = read_spectra(EXAMPLE)["masked_and_bad_pixels"].visits[0]
    assert found.mask.dtype == bool and found.wavelength.dtype == np.float64


def test_a_transient_or_a_table_gives_the_result_of_its_arrays_and_prints_its_card(model):
    transient = read_spectra(EXAMPLE)["sn_ia"]
    alone = model.classify(*lists(transient))
    found = model.classify(Transient(transient.visits))
    assert found.class_probabilities == alone.class_probabilities
    assert np.array_equal(found.redshift_posterior, alone.redshift_posterior)
    ranked = sorted(found.class_probabilities.items(), key=lambda item: -item[1])
    assert (found.top_class, found.top_probability, found.second_class) == (*ranked[0], ranked[1][0])
    card = found.card("my transient").splitlines()
    assert str(found) == found.card() and len(card) == 6 and all(len(line) <= 79 for line in card)
    assert card[0].startswith("STRIDER · my transient") and card[0].endswith(model.name) and len(card[0]) == 79
    assert card[1] == "Class     " + " · ".join(f"{name} {100 * probability:.0f}%" for name, probability in ranked[:3])
    low, high = found.redshift_68
    assert card[2] == (
        f"Redshift  {found.redshift:.2f} ({100 * found.redshift_probability:.0f}% within 0.1) · 68% range "
        f"{low:.2f}–{high:.2f}"
    )
    assert card[3] == f"Match     {found.match_strength} strength" and found.warnings == []
    assert card[4] == f"Data      11 visits over {found.days_observed:.0f} days · S/N {found.signal_to_noise:.1f}"
    assert [flag for flag, raised in found.flags.items() if raised] == ["weak_source"]
    assert card[5] == f"Checks    weak source ({100 * found.source_probability:.0f}%)"
    assert not any(mark in found.card() for mark in ("─", "█", "redshift peak"))
    sure = replace(
        found, class_probabilities=dict.fromkeys(model.classes, 0.0) | {"Ia": 0.98, "IIP": 0.012, "IIb": 0.008}
    )
    assert sure.card().splitlines()[1] == "Class     Ia 98% · IIP 1%"
    assert repr(found) == f"<Classification {ranked[0][0]} {100 * ranked[0][1]:.0f}%, z {found.redshift:.2f}>"
    assert 0.0 < found.redshift_probability <= 1.0 + 1e-12
    with pytest.raises(ValueError, match="give a table with one row per pixel"):
        model.classify(lists(transient)[0])

    hosted = model.classify(Transient(transient.visits, float(model.redshift_grid[8]), 0.01))
    assert hosted.redshift_posterior[8] > 0.9  # 5% is left for the host being wrong
    assert (
        model.classify(
            Transient(transient.visits, 0.1, 0.01),
            host_redshift=float(model.redshift_grid[8]),
            host_redshift_error=0.01,
        ).class_probabilities
        == hosted.class_probabilities
    )


def test_a_class_left_out_keeps_its_redshift_posterior_and_one_without_any_has_none(model):
    """KN had no training transients below z 1."""
    result = model.classify(strider.load_example(), classes=["Ia", "IIP"], redshift_range=(0.0, 0.8))
    rows = dict(zip(model.classes, result.class_redshift_posterior))
    assert result.class_probabilities["Ib"] == 0.0 and rows["Ib"].sum() == pytest.approx(1.0)
    assert np.isnan(rows["KN"]).all() and np.isnan(result.class_redshifts["KN"])


def test_the_redshift_given_a_class_is_its_own_posterior(model):
    result = model.classify(strider.load_example())
    conditional = result.class_redshift_posterior
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        np.array(list(result.class_probabilities.values())) @ conditional,
        result.redshift_posterior,
        rtol=1e-9,
        atol=1e-15,
    )
    ia = result.redshift_given("Ia")
    assert np.array_equal(ia.redshift_posterior, conditional[0])
    assert ia.redshift_95[0] <= ia.redshift_68[0] <= ia.redshift_68[1] <= ia.redshift_95[1]
    assert (
        ia.redshift
        == result.class_redshifts["Ia"]
        == posterior.class_redshifts(result.redshift_grid, conditional[:1], result.redshift_cell_widths)[0]
    )
    with pytest.raises(ValueError, match="unknown class 'Ib/c'"):
        result.redshift_given("Ib/c")


def test_priors_given_by_the_user(model):
    transient = strider.load_example()
    base = model.classify(transient)
    assert model.classify(transient, classes=model.classes).class_probabilities == base.class_probabilities
    assert np.array_equal(
        model.classify(transient, redshift_range=(-np.inf, np.inf)).redshift_posterior, base.redshift_posterior
    )

    kept = model.classify(transient, classes=["Ia", "Iax", "91bg"])
    shares = np.array([base.class_probabilities[name] for name in ("Ia", "Iax", "91bg")])
    np.testing.assert_allclose(
        [kept.class_probabilities[name] for name in ("Ia", "Iax", "91bg")], shares / shares.sum(), rtol=1e-12
    )
    assert sum(value for name, value in kept.class_probabilities.items() if name not in ("Ia", "Iax", "91bg")) == 0
    np.testing.assert_allclose(
        kept.redshift_given("Ia").redshift_posterior, base.redshift_given("Ia").redshift_posterior, rtol=1e-12
    )

    weights = dict.fromkeys(model.classes, 1.0)
    equal = model.classify(transient, class_prior=weights)
    np.testing.assert_allclose(
        list(equal.class_probabilities.values()), list(base.equal_prior_probabilities.values()), rtol=1e-12
    )
    ranged = model.classify(transient, redshift_range=(0.5, 1.5))
    edges = np.concatenate([[0.0], 0.5 * (model.redshift_grid[1:] + model.redshift_grid[:-1]), [3.0]])
    assert ranged.redshift_posterior[(edges[1:] <= 0.5) | (edges[:-1] >= 1.5)].sum() == 0.0
    assert 0.5 <= ranged.redshift <= 1.5
    density = np.exp(-0.5 * ((model.redshift_grid - 1.0) / 0.2) ** 2)
    expected = base.redshift_posterior * density
    np.testing.assert_allclose(
        model.classify(transient, redshift_prior=density).redshift_posterior,
        expected / expected.sum(),
        rtol=1e-9,
        atol=1e-15,
    )

    for priors, message in [
        ({"classes": ["Ia", "Iz"]}, "must be some of this model's classes: Ia, 91bg"),
        ({"class_prior": {"Ia": 0.9}}, "class_prior must give a value for each class allowed"),
        ({"classes": ["Ia"], "class_prior": {"Ia": 0.0}}, "with some above 0"),
        ({"redshift_range": (1.5, 0.5)}, "lowest first"),
        ({"redshift_range": (5.0, 6.0)}, "outside the model's redshifts, 0 to 3"),
        ({"redshift_prior": np.ones(3)}, "at each of the 12 redshifts"),
    ]:
        with pytest.raises(ValueError, match=message):
            model.classify(transient, **priors)
    with pytest.raises(ValueError, match="must be some of this model's classes"):
        model.classify_many({"one": transient}, classes=["Iz"])
    batch = model.classify_many({"one": transient}, classes=["Ia", "Iax", "91bg"])
    assert batch["one"].class_probabilities == kept.class_probabilities


def test_the_test_model_is_named(model):
    assert strider.load_model("test").fingerprint == model.fingerprint


def test_the_card_says_when_no_class_was_trained_at_the_redshift(model):
    import dataclasses

    result = model.classify(strider.load_example())
    calm = dataclasses.replace(result, flags=dict.fromkeys(result.flags, False))
    assert [line[:10].rstrip() for line in calm.card("x").splitlines()[1:]] == ["Class", "Redshift", "Match", "Data"]
    flags = {**calm.flags, "few_classes_trained": True}
    none = dataclasses.replace(calm, trained_classes=[], flags=flags)
    assert none.card("x").splitlines()[5:] == ["Checks    no class trained here"]
    few = dataclasses.replace(calm, trained_classes=["Ia", "Iax"], flags=flags)
    assert few.card("x").splitlines()[5:] == ["Checks    trained here: Ia, Iax only"]
    every = dataclasses.replace(few, flags=dict.fromkeys(result.flags, True), untrained_share=0.12)
    assert every.card("x").splitlines()[5:] == [
        f"Checks    weak source ({100 * result.source_probability:.0f}%) · trained here: Ia, Iax only",
        "          untrained subtypes 12% · brightness failed, spectra alone",
    ]


def test_the_card_lists_the_warnings(model):
    import dataclasses

    result = model.classify(strider.load_example())
    warned = dataclasses.replace(result, warnings=["second redshift near z 1.20", "spectra and brightness disagree"])
    card = warned.card("x").splitlines()
    assert card[3] == f"Match     {result.match_strength} strength · second redshift near z 1.20"
    assert card[4] == "          spectra and brightness disagree"
    assert card[5].startswith("Data")
    assert max(map(len, card)) <= 79


def test_the_card_on_one_night_with_a_long_name_and_no_signal(model):
    transient = read_spectra(EXAMPLE)["sn_ia"]
    data = lists(transient)
    data[3] = [60000.0, 60000.2] + list(data[3][2:])
    found = model.classify(*(part[:2] for part in data))
    found.signal_to_noise = -0.4
    card = found.card("a very long name that would otherwise run straight into the model's own name").splitlines()
    assert all(len(line) <= 79 for line in card) and card[0].endswith(model.name) and "…" in card[0]
    assert card[4] == "Data      2 visits on one night · S/N 0.0"


def test_classify_many_refuses_two_outputs_in_one_file(model, tmp_path):
    same = tmp_path / "out.json"
    with pytest.raises(ValueError, match="different files"):
        model.classify_many(read_spectra(EXAMPLE), same, evidence_maps=same)
    assert not same.exists()


def test_a_killed_batch_keeps_its_parquet_rows_and_a_finished_one_joins_them(model, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from strider.io import ResultsFile

    objects = read_spectra(EXAMPLE)
    results = model.classify_many(objects, keep=True)
    path = tmp_path / "out.parquet"
    writer = ResultsFile(path, model)
    for key, result in results.items():
        writer.write({key: result})  # a batch killed now has no close(): its rows are in the parts
    assert pq.read_table(path.with_name("out.parquet.parts")).num_rows == len(objects)
    writer.close()
    assert pq.read_table(path).column("object_id").to_pylist() == list(objects)
    assert not path.with_name("out.parquet.parts").exists()
    assert len(set(pq.read_table(path).column("package_sha256").to_pylist())) == 1


def test_keep_false_returns_only_the_failures(model, tmp_path):
    objects = read_spectra(EXAMPLE)
    results = model.classify_many(objects, tmp_path / "out.csv", keep=False)
    assert all(isinstance(result, str) for result in results.values())


def test_a_package_file_can_be_pinned_by_its_sha256(tmp_path):
    import hashlib as h

    digest = h.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert strider.load_model(ARCHIVE, sha256=digest).files_sha256
    with pytest.raises(ValueError, match="not the"):
        strider.load_model(ARCHIVE, sha256="0" * 64)


@pytest.mark.parametrize("bounds", [(0.5, 0.52), (0.3, 0.6), (1.0, 1.003)])
def test_a_redshift_range_keeps_every_redshift_summary_inside_it(model, bounds):
    result = model.classify(strider.load_example(), redshift_range=bounds)
    low, high = bounds
    values = [result.redshift, *result.redshift_68, *result.redshift_95, *(z for z, _ in result.redshift_peaks)]
    ia = result.redshift_given("Ia")
    values += [ia.redshift, *ia.redshift_68, *ia.redshift_95]
    assert all(low <= v <= high for v in values if np.isfinite(v)), (bounds, values)
    assert result.redshift_range == bounds


def test_without_a_range_the_summaries_are_unchanged(model):
    from strider import posterior as post

    result = model.classify(strider.load_example())
    grid, mass = result.redshift_grid, result.redshift_posterior
    assert result.redshift_range is None
    assert result.redshift_68 == post.central_interval(grid, mass, 0.68)


def test_a_class_restriction_keeps_its_redshift_interval_where_it_has_probability(model):
    result = model.classify(strider.load_example(), classes=["IIP"])
    grid, mass = result.redshift_grid, result.redshift_posterior
    support = grid[mass > 0]
    assert support.min() - 0.01 <= result.redshift_95[0] <= result.redshift_95[1] <= support.max() + 0.01


def test_the_first_spectra_follow_the_scaled_order_of_visits_on_one_date(model):
    wavelength = np.geomspace(7400, 18300, 20)
    error = np.full(20, 3.8272171853349986)
    flux = np.random.default_rng(14).normal(0, 2 * error, (2, 20))
    flux[0, 0] = 0.005740825778002498
    transient = Transient([Visit(wavelength, f, error, 60000.0) for f in flux])
    full = model.classify(transient, error_scale="auto")
    first = model.classify_first(transient, steps=[1, 2], error_scale="auto")
    assert first[1].visits_used == full.visits_used[:1]
    assert first[2].p_ia == full.p_ia


def test_the_first_spectra_check_priors_before_the_spectra(model):
    unusable = Transient([Visit(np.geomspace(7400, 18300, 20), np.full(20, np.nan), np.ones(20), 60000.0)])
    with pytest.raises(ValueError, match="host"):
        model.classify_first(unusable, host_redshift=0.5)
