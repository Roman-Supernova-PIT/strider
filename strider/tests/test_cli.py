import json
import signal

import pytest

from strider import cli, registry
from strider.cli import main
from strider.package import read_package
from strider.tests.synthetic_package import ARCHIVE


@pytest.mark.parametrize("argv", [[], ["classify"], ["models"], ["check"], ["check-model"]])
def test_help(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--help"])
    assert excinfo.value.code == 0
    assert "usage: strider" in capsys.readouterr().out


def test_models(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STRIDER_MODEL_DIR", str(tmp_path))
    main(["models"])
    assert "strider-15class" in capsys.readouterr().out
    monkeypatch.setattr(registry, "MODELS", {})
    main(["models"])
    assert "No models released yet." in capsys.readouterr().out
    monkeypatch.setitem(registry.MODELS, "m", {"url": "", "sha256": ""})
    (tmp_path / "m.tar.gz").touch()
    main(["models"])
    assert "m  cached" in capsys.readouterr().out


@pytest.mark.parametrize(
    "model, message",
    [("./no/such/package.tar.gz", "no model package at"), ("no-such-model", "unknown model 'no-such-model'")],
)
def test_a_missing_model_is_a_short_error(model, message):
    pytest.importorskip("torch")
    with pytest.raises(SystemExit, match=message):
        main(["check", "--model", model])


def test_check_prints_versions_and_timing_and_compares_the_stored_results(monkeypatch, tmp_path, capsys):
    pytest.importorskip("torch")
    main(["check", "--model", str(ARCHIVE)])
    out = capsys.readouterr().out
    assert "PyTorch" in out and "s per object" in out and "The example matches its stored results." in out

    stored = json.loads(cli.EXPECTED.read_text())
    entry = stored[read_package(ARCHIVE).info["fingerprint"]]
    entry["sn_ia"]["class_probabilities"][0] += 0.01
    changed = tmp_path / "expected.json"
    changed.write_text(json.dumps(stored))
    monkeypatch.setattr(cli, "EXPECTED", changed)
    with pytest.raises(SystemExit) as excinfo:
        main(["check", "--model", str(ARCHIVE)])
    assert excinfo.value.code == 1
    assert "sn_ia: class_probabilities" in capsys.readouterr().out
    changed.write_text("{}")
    main(["check", "--model", str(ARCHIVE)])
    assert "only the timing is checked" in capsys.readouterr().out


def test_classify_writes_one_row_per_object(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    output = tmp_path / "results.parquet"
    main(["classify", str(cli.EXAMPLE), "-o", str(output), "--model", str(ARCHIVE)])
    assert "Classified 3 of 3 objects" in capsys.readouterr().out
    assert pq.read_table(output).num_rows == 3
    main(["classify", str(cli.EXAMPLE), "-o", str(output), "--model", str(ARCHIVE), "--device", "cpu", "--batch", "2"])
    assert "Classified 3 of 3 objects" in capsys.readouterr().out
    stop = signal.signal(signal.SIGTERM, signal.SIG_DFL)  # classify set it: a scheduler's stop exits through Python
    with pytest.raises(SystemExit, match="143"):
        stop(signal.SIGTERM, None)


def test_classify_checks_its_input_and_output_first_and_fails_when_nothing_is_classified(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    with pytest.raises(SystemExit, match="cannot write .*nowhere"):
        main(["classify", str(cli.EXAMPLE), "-o", str(tmp_path / "nowhere" / "out.parquet"), "--model", str(ARCHIVE)])
    rows = [
        row | {"wavelength": row["wavelength"][:20], "flux": row["flux"][:20], "error": row["error"][:20], "mask": None}
        for row in pq.read_table(cli.EXAMPLE).to_pylist()
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "short.parquet")
    with pytest.raises(SystemExit, match="no object could be classified"):
        main(
            ["classify", str(tmp_path / "short.parquet"), "-o", str(tmp_path / "out.parquet"), "--model", str(ARCHIVE)]
        )
    assert pq.read_table(tmp_path / "out.parquet").num_rows == 3


def test_without_a_released_model_the_error_names_the_option(monkeypatch):
    monkeypatch.setattr(registry, "MODELS", {})
    with pytest.raises(SystemExit, match="give --model test for the bundled test model"):
        main(["check"])


def test_a_download_failure_gives_a_hint(monkeypatch, tmp_path):
    pytest.importorskip("torch")

    def offline(name=None):
        raise ConnectionError("no route to host")

    monkeypatch.setitem(registry.MODELS, "m", {"url": "", "sha256": ""})
    monkeypatch.setattr(registry, "fetch", offline)
    with pytest.raises(SystemExit, match="no route to host. If a model was being downloaded, check the network"):
        main(["check", "--model", "m"])


def test_models_prints_a_card(capsys):
    main(["models", "--card", str(ARCHIVE)])
    assert capsys.readouterr().out.startswith("# strider-synthetic-test (version 1.0)")


def test_classify_one_transient_prints_its_card_and_takes_the_priors(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    import strider
    from strider.io import read_transient

    csv = cli.EXAMPLE.parent / "example.csv"
    model = strider.load_model("test")
    main(["classify", str(csv), "--model", "test"])
    assert capsys.readouterr().out == model.classify(read_transient(csv)).card("example") + "\n"
    main(
        [
            "classify",
            str(csv),
            "--model",
            "test",
            "--classes",
            "Ia,Iax",
            "--zmin",
            "0.5",
            "--class-prior",
            "Ia=0.5,Iax=0.5",
            "-o",
            str(tmp_path / "out.json"),
        ]
    )
    assert "Classified 1 of 1 objects" in capsys.readouterr().out
    found = json.loads((tmp_path / "out.json").read_text())["example"]
    expected = model.classify(
        read_transient(csv),
        classes=["Ia", "Iax"],
        class_prior={"Ia": 0.5, "Iax": 0.5},
        redshift_range=(0.5, float("inf")),
    )
    assert found["class_probabilities"] == expected.class_probabilities
    main(["classify", str(cli.EXAMPLE), "--model", "test", "-o", str(tmp_path / "out.parquet"), "--zmax", "1"])
    assert pq.read_table(tmp_path / "out.parquet").column("redshift_95_upper").to_pylist()[0] <= 1.2
    with pytest.raises(SystemExit, match="give -o a file ending in .parquet, .csv, .json"):
        main(["classify", str(csv), "--model", "test", "-o", str(tmp_path / "out.txt")])
    with pytest.raises(SystemExit) as excinfo:
        main(["classify", str(csv), "--class-prior", "Ia"])
    assert excinfo.value.code == 2 and "not a list of CLASS=VALUE" in capsys.readouterr().err


def test_classify_takes_an_error_scale(tmp_path, capsys):
    pytest.importorskip("torch")
    import numpy as np

    import strider
    from strider.io import read_transient

    # the example with white noise added, scattering 1.5 times its errors, so "auto" has something to find
    lines = (cli.EXAMPLE.parent / "example.csv").read_text().splitlines()
    rng = np.random.default_rng(7)
    rows = [line.split(",") for line in lines[1:]]
    for row in rows:
        row[2] = repr(float(row[2]) + float(np.sqrt(1.25) * float(row[3]) * rng.normal()))
    csv = tmp_path / "noisy.csv"
    csv.write_text("\n".join([lines[0], *(",".join(row) for row in rows)]) + "\n")
    model = strider.load_model("test")
    plain = model.classify(read_transient(csv))
    for flag, value in (("1.5", 1.5), ("auto", "auto")):
        out = tmp_path / f"{flag}.json"
        main(["classify", str(csv), "--model", "test", "--error-scale", flag, "-o", str(out)])
        expected = model.classify(read_transient(csv), error_scale=value)
        found = json.loads(out.read_text())["noisy"]
        assert found["class_probabilities"] == expected.class_probabilities
        assert found["error_scale"] == expected.error_scale > 1.2 and expected.p_ia != plain.p_ia
    with pytest.raises(SystemExit, match="error_scale"):
        main(["classify", str(csv), "--model", "test", "--error-scale", "0"])
    with pytest.raises(SystemExit) as excinfo:
        main(["classify", str(csv), "--model", "test", "--error-scale", "AUTO"])
    assert excinfo.value.code == 2 and 'error scale must be "auto" or a number' in capsys.readouterr().err


def test_classify_prints_each_card_and_writes_figures_and_evidence_maps(tmp_path, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    h5py = pytest.importorskip("h5py")
    import matplotlib

    matplotlib.use("Agg")
    main(
        [
            "classify",
            str(cli.EXAMPLE),
            "--model",
            "test",
            "--plot",
            str(tmp_path / "figures"),
            "--evidence-maps",
            str(tmp_path / "evidence.h5"),
        ]
    )
    out = capsys.readouterr().out
    assert out.count("STRIDER · ") == 3 and "STRIDER · masked_and_bad_pixels" in out
    assert f"Figures in {tmp_path / 'figures'}" in out and f"Evidence maps in {tmp_path / 'evidence.h5'}" in out
    assert len(list((tmp_path / "figures").glob("*.png"))) == 3
    with h5py.File(tmp_path / "evidence.h5") as found:
        assert found["log_evidence"].shape[0] == 3
    with pytest.raises(SystemExit, match="--evidence-maps would overwrite the input"):
        main(["classify", str(cli.EXAMPLE), "--model", "test", "--evidence-maps", str(cli.EXAMPLE)])


def test_classify_refuses_two_outputs_in_one_file(tmp_path):
    pytest.importorskip("torch")
    same = tmp_path / "out.json"
    with pytest.raises(SystemExit, match="different files"):
        main(["classify", str(cli.EXAMPLE), "-o", str(same), "--evidence-maps", str(same), "--model", str(ARCHIVE)])
    assert not same.exists()
