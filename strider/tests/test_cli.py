import json
import signal
import sys

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
    fingerprint = read_package(ARCHIVE).info["fingerprint"]
    entry = stored[fingerprint]
    original = json.loads(json.dumps(entry))
    assert set(entry) == {"model", "fast", "full"}
    main(["check", "--model", str(ARCHIVE), "--full-scan"])
    out = capsys.readouterr().out
    assert "full scan" in out and "The example matches its stored results." in out
    entry["fast"]["sn_ia"]["class_probabilities"][0] += 0.01
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
    changed.write_text(json.dumps({fingerprint: {"model": original["model"], "full": original["full"]}}))
    with pytest.raises(SystemExit):  # an entry without the scan asked for is a problem, not a pass
        main(["check", "--model", str(ARCHIVE)])
    assert "no stored results for the fast scan" in capsys.readouterr().out
    changed.write_text(json.dumps({fingerprint: {"model": original["model"]} | original["full"]}))  # 0.1's layout
    main(["check", "--model", str(ARCHIVE), "--full-scan"])
    assert "The example matches its stored results." in capsys.readouterr().out


def test_classify_writes_one_row_per_object(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    output = tmp_path / "results.parquet"
    main(["classify", str(cli.EXAMPLE), "-o", str(output), "--model", str(ARCHIVE)])
    assert "Classified 3 of 3 objects" in capsys.readouterr().out
    assert pq.read_table(output).num_rows == 3
    assert set(pq.read_table(output).column("brightness_scan").to_pylist()) == {"fast"}
    with pytest.raises(SystemExit, match="results.parquet exists already; write to another file"):
        main(["classify", str(cli.EXAMPLE), "-o", str(output), "--model", str(ARCHIVE)])
    batched, full = tmp_path / "batched.parquet", tmp_path / "full.parquet"
    main(["classify", str(cli.EXAMPLE), "-o", str(batched), "--model", str(ARCHIVE), "--device", "cpu", "--batch", "2"])
    assert "Classified 3 of 3 objects" in capsys.readouterr().out
    main(["classify", str(cli.EXAMPLE), "-o", str(full), "--model", str(ARCHIVE), "--full-scan"])
    assert set(pq.read_table(full).column("brightness_scan").to_pylist()) == {"full"}
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


def test_classify_one_transient_prints_its_report_and_takes_the_priors(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    import strider
    from strider.io import read_transient

    csv = cli.EXAMPLE.parent / "example.csv"
    model = strider.load_model("test")
    from strider import terminal

    main(["classify", str(csv), "--model", "test"])
    result = model.classify(read_transient(csv))
    report = terminal.report(result, "example", terminal.Style(sys.stdout))
    assert capsys.readouterr().out == report + "\n"
    assert "\033[" not in report and "Class     " + result.report().splitlines()[1][10:] in report  # plain
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


def test_classify_prints_each_report_and_writes_figures_and_evidence_maps(tmp_path, capsys):
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
    assert out.count("   Class     ") == 3 and "── masked_and_bad_pixels ─" in out
    assert out.count(f"Figure    {tmp_path / 'figures'}/") == 3 and "Figures saved in" not in out  # on each report
    assert f"Evidence maps in {tmp_path / 'evidence.h5'}" in out
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


def test_classify_reads_a_table_with_its_own_column_names_and_units(tmp_path, capsys):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    rows = [
        {"SNID": row["object_id"], **row, "wavelength": [w / 1e4 for w in row["wavelength"]]}
        for row in pq.read_table(cli.EXAMPLE).to_pylist()
    ]
    pq.write_table(
        pa.Table.from_pylist([{k: v for k, v in row.items() if k != "object_id"} for row in rows]),
        tmp_path / "own.parquet",
    )
    main(["classify", str(cli.EXAMPLE), "-o", str(tmp_path / "plain.parquet"), "--model", str(ARCHIVE)])
    own = ["--columns", "object_id=SNID", "--wavelength-unit", "micron"]
    main(
        [
            "classify",
            str(tmp_path / "own.parquet"),
            "-o",
            str(tmp_path / "own_out.parquet"),
            "--model",
            str(ARCHIVE),
            *own,
        ]
    )
    assert "Classified 3 of 3 objects" in capsys.readouterr().out
    plain, found = (pq.read_table(tmp_path / name).to_pylist() for name in ("plain.parquet", "own_out.parquet"))
    assert [row["object_id"] for row in found] == [row["object_id"] for row in plain]
    assert [row["p_ia"] for row in found] == pytest.approx([row["p_ia"] for row in plain], abs=1e-9)
    with pytest.raises(SystemExit, match="are for a parquet table of spectra"):
        main(["classify", str(cli.EXAMPLE.with_suffix(".csv")), "--model", str(ARCHIVE), *own])
    with pytest.raises(SystemExit, match="has no column for object_id=SNID"):
        main(["classify", str(cli.EXAMPLE), "--model", str(ARCHIVE), "--columns", "object_id=SNID"])


def test_jobs_of_one_input_resume_and_merge_on_the_command_line(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    base = ["classify", str(cli.EXAMPLE), "--model", str(ARCHIVE)]
    with pytest.raises(SystemExit, match="--job and --resume need a results file"):
        main([*base, "--job", "0/2"])
    with pytest.raises(SystemExit, match="a job K/N needs N of at least 1 and K from 0 to N - 1, not 2/2"):
        main([*base, "-o", str(tmp_path / "bad.parquet"), "--job", "2/2"])
    for k in (0, 1):
        main([*base, "-o", str(tmp_path / f"job{k}.parquet"), "--job", f"{k}/2"])
    jobs = [str(tmp_path / f"job{k}.parquet") for k in (0, 1)]
    main(["merge", *jobs, "-o", str(tmp_path / "all.parquet")])
    assert "Joined 2 files" in capsys.readouterr().out
    assert pq.read_table(tmp_path / "all.parquet").num_rows == 3
    record = json.loads(pq.read_schema(tmp_path / "all.parquet").metadata[b"strider_run"])
    assert record["input"]["path"] == str(cli.EXAMPLE.resolve()) and len(record["input"]["sha256"]) == 64
    with pytest.raises(SystemExit, match="are not each of one run's once"):
        main(["merge", jobs[0], "-o", str(tmp_path / "half.parquet")])


def test_info_describes_the_install_and_the_run_behind_a_results_file(tmp_path, capsys):
    pytest.importorskip("torch")
    main(["info"])
    shown = capsys.readouterr().out
    assert shown.startswith("STRIDER ") and "Packages: numpy " in shown and "Models in " in shown
    main(["info", "--json"])
    assert set(json.loads(capsys.readouterr().out)["packages"]) >= {"numpy", "torch", "scipy", "pyarrow"}
    output = tmp_path / "out.parquet"
    main(["classify", str(cli.EXAMPLE), "-o", str(output), "--model", str(ARCHIVE)])
    capsys.readouterr()
    main(["info", str(output)])
    shown = capsys.readouterr().out
    assert shown.startswith("Complete run of strider-synthetic-test-1.0") and "Objects: 3 (" in shown
    assert "Seconds per object: median " in shown
    main(["info", str(output), "--json"])
    assert json.loads(capsys.readouterr().out)["summary"]["objects"] == 3


def test_info_works_without_torch():
    import subprocess
    import sys

    code = "import sys; sys.modules['torch'] = None; import strider.cli; strider.cli.main(['info'])"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "Packages: numpy" in run.stdout


def test_a_saved_transient_reruns_and_says_if_its_result_differs(tmp_path, capsys):
    pytest.importorskip("torch")
    base = ["classify", str(cli.EXAMPLE), "--model", str(ARCHIVE)]
    with pytest.raises(SystemExit, match="--save-transient saves a folder for each transient: name them with --only"):
        main([*base, "--save-transient", str(tmp_path / "b")])
    with pytest.raises(SystemExit, match="has no object nowhere"):
        main([*base, "--only", "sn_ia,nowhere"])
    kept = tmp_path / "saved"
    main([*base, "--only", "sn_ia,a_masked_block", "--save-transient", str(kept), "--light-curve-errors", "correct"])
    assert "Saved in" in capsys.readouterr().out
    from strider.saved import saved_folder

    assert sorted(p.name.rsplit("-", 1)[0] for p in kept.iterdir()) == ["a_masked_block", "sn_ia"]
    assert {p.name for p in saved_folder(kept, "sn_ia").iterdir()} == {"spectra.parquet", "record.json", "result.json"}
    main(["rerun", str(kept), "--model", str(ARCHIVE)])
    assert capsys.readouterr().out.splitlines() == ["a_masked_block: the same", "sn_ia: the same"]
    saved = json.loads((saved_folder(kept, "sn_ia") / "result.json").read_text())
    (saved_folder(kept, "sn_ia") / "result.json").write_text(json.dumps(saved | {"p_ia": saved["p_ia"] + 0.1}))
    with pytest.raises(SystemExit, match="1 of 1 saved transients differ"):
        main(["rerun", str(saved_folder(kept, "sn_ia")), "--model", str(ARCHIVE)])
    assert "p_ia: " in capsys.readouterr().out


def test_the_example_in_each_pipeline_layout_gives_the_same_results(tmp_path, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")
    data = cli.EXAMPLE.parent
    own = ["--columns", "object_id=SNID,wavelength=lam,error=flux_error", "--wavelength-unit", "micron"]
    runs = {
        "plain": [str(cli.EXAMPLE)],
        "by_pixel": [str(data / "example_by_pixel.parquet")],
        "own_names": [str(data / "example_own_names.parquet"), *own, "--flux-density", "f_nu"],
    }
    found = {}
    for name, given in runs.items():
        main(["classify", given[0], "-o", str(tmp_path / f"{name}.parquet"), "--model", str(ARCHIVE), *given[1:]])
        assert "Classified 3 of 3 objects" in capsys.readouterr().out
        found[name] = {row["object_id"]: row for row in pq.read_table(tmp_path / f"{name}.parquet").to_pylist()}
    # the same numbers one row per pixel; converted units differ in the last bits, which float32 can round apart
    for name, tolerance in (("by_pixel", 0.0), ("own_names", 1e-5)):
        assert found[name].keys() == found["plain"].keys()
        for key, row in found["plain"].items():
            assert found[name][key]["p_ia"] == pytest.approx(row["p_ia"], abs=tolerance), (name, key)
            assert found[name][key]["redshift"] == pytest.approx(row["redshift"], abs=tolerance), (name, key)
            assert found[name][key]["top_class"] == row["top_class"], (name, key)


def test_a_saved_transient_reruns_with_the_model_it_names(tmp_path, capsys):
    pytest.importorskip("torch")
    kept = tmp_path / "saved"
    base = ["classify", str(cli.EXAMPLE), "--model", registry.TEST_MODEL]
    main([*base, "--only", "sn_ia", "--save-transient", str(kept)])
    capsys.readouterr()
    main(["rerun", str(kept)])
    assert capsys.readouterr().out.splitlines() == ["sn_ia: the same"]


def test_rerun_finds_a_change_in_any_part_of_a_result(tmp_path, capsys):
    pytest.importorskip("torch")
    from strider.saved import rerun, saved_folder
    from strider.model import load_model

    kept = tmp_path / "saved"
    main(["classify", str(cli.EXAMPLE), "--model", str(ARCHIVE), "--only", "sn_ia", "--save-transient", str(kept)])
    capsys.readouterr()
    folder = saved_folder(kept, "sn_ia")
    saved = json.loads((folder / "result.json").read_text())
    model = load_model(ARCHIVE)
    for change, field in (
        (lambda r: r["redshift_posterior"].__setitem__(5, r["redshift_posterior"][5] + 0.01), "redshift_posterior"),
        (lambda r: r.__setitem__("redshift_68", [r["redshift_68"][0] + 0.01, r["redshift_68"][1]]), "redshift_68"),
        (lambda r: r.__setitem__("visits_used", r["visits_used"][1:]), "visits_used"),
        (lambda r: r["light_curve_check"].__setitem__("status", "corrected"), "light_curve_check.status"),
    ):
        changed = json.loads(json.dumps(saved))
        change(changed)
        (folder / "result.json").write_text(json.dumps(changed))
        differences, _ = rerun(folder, model)
        assert any(line.startswith(field) for line in differences), (field, differences)
    (folder / "result.json").write_text(json.dumps(saved))
    assert rerun(folder, model)[0] == []
    slashed = saved_folder(kept, "a/b")
    assert len({slashed, saved_folder(kept, "a_b"), saved_folder(kept, slashed.name)}) == 3  # distinct ids, folders
    with pytest.raises(SystemExit, match="exist already; give --save-transient another folder"):
        main(["classify", str(cli.EXAMPLE), "--model", str(ARCHIVE), "--only", "sn_ia", "--save-transient", str(kept)])
