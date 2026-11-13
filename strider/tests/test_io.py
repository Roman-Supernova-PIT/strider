"""Reading one transient from a table, a CSV or ECSV file or SMDC files, and writing results by the file's ending."""

import json
from importlib.resources import files

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")

import strider  # noqa: E402
from strider.io import (
    Transient,
    read_smdc,
    read_transient,
    write_evidence_maps,
    write_redshift_curves,  # noqa: E402
    write_results,
)
from strider.tests.synthetic_package import ARCHIVE  # noqa: E402

CSV = files("strider") / "data" / "example.csv"


@pytest.fixture(scope="module")
def model():
    return strider.load_model(ARCHIVE)


def pixel_rows(transient):
    """The transient as columns with one row per pixel."""
    visits = transient.visits
    return {
        "mjd": np.concatenate([np.full(len(visit.flux), visit.mjd) for visit in visits]),
        **{
            name: np.concatenate([getattr(visit, name) for visit in visits]) for name in ("wavelength", "flux", "error")
        },
    }


def same_visits(found, expected):
    return len(found.visits) == len(expected.visits) and all(
        a.mjd == b.mjd and all(np.array_equal(getattr(a, name), getattr(b, name)) for name in ("wavelength", "flux"))
        for a, b in zip(found.visits, expected.visits)
    )


def test_the_example_is_one_transient_or_a_batch_and_the_csv_holds_the_same_transient():
    transient = strider.load_example()
    assert isinstance(transient, Transient) and len(transient.visits) == 11
    assert list(strider.load_example(batch=True)) == ["sn_ia", "masked_and_bad_pixels", "a_masked_block"]
    assert same_visits(read_transient(CSV), transient)


def test_a_table_takes_the_common_column_names_and_groups_rows_into_visits():
    transient = strider.load_example()
    rows = pixel_rows(transient)
    renamed = {"TIME": rows["mjd"], "Wavelength": rows["wavelength"], "flux": rows["flux"], "fluxerr": rows["error"]}
    found = Transient.from_table(
        renamed
        | {"host_redshift": np.full(len(rows["mjd"]), 0.5), "host_redshift_error": np.full(len(rows["mjd"]), 0.01)}
    )
    assert same_visits(found, transient) and np.array_equal(found.visits[3].error, transient.visits[3].error)
    assert (found.host_redshift, found.host_redshift_error) == (0.5, 0.01)
    with pytest.raises(ValueError, match="the table has no error column; its columns are TIME, Wavelength, flux"):
        Transient.from_table({key: value for key, value in renamed.items() if key != "fluxerr"})
    with pytest.raises(ValueError, match="repeat a wavelength; give each visit its own MJD"):
        Transient.from_table(renamed | {"TIME": np.full(len(rows["mjd"]), 60000.0)})
    with pytest.raises(ValueError, match="gives 2 values of host_redshift"):
        Transient.from_table(renamed | {"host_redshift": np.arange(len(rows["mjd"])) % 2})


def test_pandas_and_astropy_tables_with_units():
    pandas = pytest.importorskip("pandas")
    units = pytest.importorskip("astropy.units")
    from astropy.table import Table

    transient = strider.load_example()
    rows = pixel_rows(transient)
    assert same_visits(Transient.from_table(pandas.DataFrame(rows).rename(columns={"error": "flux_err"})), transient)
    table = Table(rows)
    table["wavelength"].unit = units.micron
    table["wavelength"] /= 1e4
    table["flux"].unit = table["error"].unit = units.erg / units.s / units.cm**2 / units.AA
    found = Transient.from_table(table)
    np.testing.assert_allclose(found.visits[0].wavelength, transient.visits[0].wavelength, rtol=1e-12)
    np.testing.assert_allclose(found.visits[0].flux, transient.visits[0].flux, rtol=1e-12)
    from astropy.constants import c

    fnu = (units.Quantity(rows["flux"], table["flux"].unit) * (rows["wavelength"] * units.AA) ** 2 / c).to(units.mJy)
    table["flux"], table["error"] = fnu, fnu * rows["error"] / rows["flux"]
    np.testing.assert_allclose(Transient.from_table(table).visits[0].flux, transient.visits[0].flux, rtol=1e-9)


def test_csv_and_ecsv_files(model, tmp_path):
    pytest.importorskip("astropy")
    from astropy.table import Table

    transient = strider.load_example()
    Table(pixel_rows(transient)).write(tmp_path / "one.ecsv")
    assert same_visits(read_transient(tmp_path / "one.ecsv"), transient)
    with pytest.raises(ValueError, match="reads a .csv or .ecsv file"):
        read_transient(tmp_path / "one.txt")
    result = model.classify(read_transient(CSV))
    assert result.class_probabilities == model.classify(transient).class_probabilities


def test_smdc_files_one_per_visit(tmp_path):
    h5py = pytest.importorskip("h5py")
    transient = strider.load_example()
    paths = []
    for index, visit in enumerate(transient.visits[:3]):
        paths.append(tmp_path / f"visit{index}.h5")
        with h5py.File(paths[-1], "w") as spectrum:
            for name, values in (("lamb", visit.wavelength), ("flam", visit.flux), ("func", visit.error)):
                spectrum[f"combined/data/{name}"] = values
    found = read_smdc(paths, [visit.mjd for visit in transient.visits[:3]], 0.5, 0.01)
    assert same_visits(found, Transient(transient.visits[:3])) and found.host_redshift == 0.5
    with pytest.raises(ValueError, match="give one MJD for each file: there are 3 files and 1 MJDs"):
        read_smdc(paths, [60000.0])
    with h5py.File(tmp_path / "other.h5", "w") as spectrum:
        spectrum["flux"] = np.ones(3)
    with pytest.raises(ValueError, match="has no combined/data/lamb"):
        read_smdc([tmp_path / "other.h5"], [60000.0])


def test_results_follow_the_file_ending(model, tmp_path):
    objects = strider.load_example(batch=True)
    results = model.classify_many(objects) | {"failed": "ValueError: no usable visit"}
    for ending in ("parquet", "csv", "json"):
        write_results(results, tmp_path / f"results.{ending}", model)
    table = pq.read_table(tmp_path / "results.parquet")
    found = json.loads((tmp_path / "results.json").read_text())
    import pyarrow.csv

    summary = pyarrow.csv.read_csv(tmp_path / "results.csv")
    assert "redshift_posterior" not in summary.column_names and summary.column_names[:2] == ["object_id", "top_class"]
    assert summary.column_names == [name for name in table.column_names if name != "redshift_posterior"]
    one = results["sn_ia"]
    row = table.to_pylist()[0]
    ia = one.redshift_given("Ia")
    assert row["top_class"] == one.top_class == found["sn_ia"]["top_class"] == summary["top_class"][0].as_py()
    assert row["second_probability"] == one.second_probability <= one.top_probability
    assert row["redshift_ia"] == ia.redshift == found["sn_ia"]["redshift_ia"]
    assert (row["redshift_ia_95_lower"], row["redshift_ia_95_upper"]) == ia.redshift_95
    assert row["redshift_probability"] == one.redshift_probability
    second = one.redshift_peaks[1] if len(one.redshift_peaks) > 1 else (None, None)
    assert (row["second_peak_redshift"], row["second_peak_probability"]) == second
    assert row["top_class_redshift"] == one.class_redshifts[one.top_class]
    assert (row["signal_to_noise"], row["days_observed"]) == (one.signal_to_noise, one.days_observed)
    assert found["sn_ia"]["redshift_peaks"][0] == {
        "redshift": one.redshift,
        "probability": one.redshift_probability,
    }
    assert (row["best_match_class"], row["best_match_redshift"]) == one.best_match
    assert found["sn_ia"]["best_match"] == {"class": one.best_match[0], "redshift": one.best_match[1]}
    assert row["match_strength"] == one.match_strength == summary["match_strength"][0].as_py()
    assert row["warnings"] == one.warnings and (summary["warnings"][0].as_py() or "") == "; ".join(one.warnings)
    assert "log_evidence" not in found["sn_ia"]
    assert len(found["sn_ia"]["redshift_posterior"]) == len(found["sn_ia"]["redshift_grid"])
    assert found["failed"]["failure"].startswith("ValueError") and summary["failure"][3].as_py().startswith("Value")
    with pytest.raises(ValueError, match="give a file ending in .parquet, .csv, .json"):
        write_results(results, tmp_path / "results.txt", model)


def test_evidence_maps_give_the_posterior_under_any_prior(model, tmp_path):
    h5py = pytest.importorskip("h5py")
    results = model.classify_many(strider.load_example(batch=True)) | {"failed": "ValueError: no usable visit"}
    write_evidence_maps(results, tmp_path / "evidence.h5", model)
    with h5py.File(tmp_path / "evidence.h5") as found:
        names = [name.decode() for name in found["object_id"]]
        evidence, prior = found["log_evidence"][:], found["log_survey_prior"][:]
        assert names == ["sn_ia", "masked_and_bad_pixels", "a_masked_block"]
        assert [name.decode() for name in found["classes"]] == model.classes and found.attrs["model"] == model.name
    one = results["sn_ia"]
    np.testing.assert_allclose(evidence[0], one.log_evidence, rtol=1e-6)
    # where the model trained, the evidence with the survey prior is the result's posterior
    joint = np.nan_to_num(one.class_redshift_posterior) * np.array(list(one.class_probabilities.values()))[:, None]
    trained = np.isfinite(prior)
    rebuilt = np.exp(evidence[0][trained] + prior[trained])
    np.testing.assert_allclose(rebuilt / rebuilt.sum(), joint[trained] / joint[trained].sum(), rtol=1e-5, atol=1e-9)
    with pytest.raises(ValueError, match="no transient was classified"):
        write_evidence_maps({"failed": "ValueError: no usable visit"}, tmp_path / "none.h5", model)


def test_every_output_records_each_transient_s_error_scale(model, tmp_path):
    h5py = pytest.importorskip("h5py")
    objects = strider.load_example(batch=True)
    scales = dict(zip(objects, (1.0, 1.5, 2.0)))
    results = {key: model.classify(transient, error_scale=scales[key]) for key, transient in objects.items()}
    results["failed"] = "ValueError: no usable visit"
    for ending in (".parquet", ".csv"):
        write_results(results, tmp_path / f"results{ending}", model)
    table = pq.read_table(tmp_path / "results.parquet")
    assert table.schema.names[-1] == "error_scale"  # new columns go at the end
    assert table.column("error_scale").to_pylist() == [*scales.values(), None]
    assert (tmp_path / "results.csv").read_text().splitlines()[0].endswith(',"error_scale"')
    assert json.loads(json.dumps(results["a_masked_block"].to_dict()))["error_scale"] == 2.0
    write_evidence_maps(results, tmp_path / "evidence.h5", model)
    with h5py.File(tmp_path / "evidence.h5") as found:
        assert found["error_scale"][:].tolist() == list(scales.values())
        assert list(found["object_id"].asstr()[:]) == list(scales)


def test_redshift_curves_in_qp(model, tmp_path):
    qp = pytest.importorskip("qp")
    results = model.classify_many(strider.load_example(batch=True))
    write_redshift_curves(results, tmp_path / "redshifts.hdf5", model)
    found = qp.read(str(tmp_path / "redshifts.hdf5"))
    assert found.npdf == 3 and list(found.ancil["object_id"]) == list(results)
    one = results["sn_ia"]
    np.testing.assert_allclose(
        found.pdf(model.redshift_grid[1:-1])[0], (one.redshift_posterior / model.widths)[1:-1], rtol=1e-6
    )


def test_a_batch_writes_its_evidence_maps_as_it_goes_and_keeps_them_if_it_stops(model, tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    objects = strider.load_example(batch=True)
    results = model.classify_many(objects, evidence_maps=tmp_path / "all.h5", chunk=1)
    with h5py.File(tmp_path / "all.h5") as found:
        assert [name.decode() for name in found["object_id"]] == list(objects)
        assert found["log_evidence"].chunks[0] == 1 and found["log_evidence"].compression == "gzip"
        np.testing.assert_allclose(found["log_evidence"][2], results["a_masked_block"].log_evidence, rtol=1e-6)
    each = model._each

    def stops(*args):
        for n, item in enumerate(each(*args)):
            if n == 2:
                raise KeyboardInterrupt
            yield item

    monkeypatch.setattr(model, "_each", stops)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, evidence_maps=tmp_path / "stopped.h5", chunk=1)
    with h5py.File(tmp_path / "stopped.h5") as found:
        assert [name.decode() for name in found["object_id"]] == list(objects)[:2]


def test_object_ids_need_not_be_text(model, tmp_path):
    h5py = pytest.importorskip("h5py")
    results = dict(enumerate(model.classify_many(strider.load_example(batch=True)).values()))
    write_results(results, tmp_path / "results.parquet", model)
    assert pq.read_table(tmp_path / "results.parquet")["object_id"].to_pylist() == ["0", "1", "2"]
    numpy_ids = {np.int64(key): result for key, result in results.items()}
    write_results(numpy_ids, tmp_path / "results.json", model)
    assert list(json.loads((tmp_path / "results.json").read_text())) == ["0", "1", "2"]
    with pytest.raises(ValueError, match="two transients have the id '0'"):
        write_results({0: results[0], "0": results[0]}, tmp_path / "twice.json", model)
    write_evidence_maps(results, tmp_path / "evidence.h5", model)
    with h5py.File(tmp_path / "evidence.h5") as found:
        assert [name.decode() for name in found["object_id"]] == ["0", "1", "2"]


def test_a_result_writes_json_without_nan_and_pickles_without_its_visits(model):
    import pickle

    result = model.classify(strider.load_example())
    result.signal_to_noise = float("nan")
    found = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert found["signal_to_noise"] is None
    copied = pickle.loads(pickle.dumps(result))
    assert len(pickle.dumps(result)) < 1_000_000 and copied.class_probabilities == result.class_probabilities
    assert not hasattr(copied, "_source") and hasattr(result, "_source")


def test_spectra_written_with_no_host_redshift_read_back(tmp_path):
    """A table where no transient has a host redshift keeps an all-empty column, which reads back as no host."""
    from strider.io import read_spectra, write_spectra

    objects = strider.load_example(batch=True)
    assert all(transient.host_redshift is None for transient in objects.values())
    write_spectra(objects, tmp_path / "spectra.parquet")
    found = read_spectra(tmp_path / "spectra.parquet")
    assert list(found) == list(objects)
    assert all(found[key].host_redshift is None and found[key].host_redshift_error is None for key in found)
    assert all(len(found[key].visits) == len(objects[key].visits) for key in found)


def test_redshift_curves_need_a_file_qp_can_write(model, tmp_path):
    from strider.io import RedshiftCurvesFile

    with pytest.raises(ValueError, match="cannot write redshift curves"):
        RedshiftCurvesFile(tmp_path / "curves.csv", model)


def test_a_fixed_size_list_column_reads_like_a_list_column():
    pa = pytest.importorskip("pyarrow")
    from strider.io import _rows

    column = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64(), 2))
    assert [row.tolist() for row in _rows(column)] == [[1.0, 2.0], [3.0, 4.0]]
