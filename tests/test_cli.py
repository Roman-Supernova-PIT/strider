import json
from pathlib import Path

import pytest

from strider.cli import build_parser, main


def test_multiple_epoch_files_accept_one_phase_each():
    args = build_parser().parse_args(
        ["classify", "epoch1.txt", "epoch2.txt", "--phase", "-7", "0"]
    )
    assert args.input == [Path("epoch1.txt"), Path("epoch2.txt")]
    assert args.phase == [-7.0, 0.0]


def test_current_model_files_accept_one_observer_time_each():
    args = build_parser().parse_args(
        ["classify", "epoch1.txt", "epoch2.txt", "--time", "62000", "62012"]
    )
    assert args.time == [62000.0, 62012.0]


def test_evidence_evolution_has_an_explicit_output():
    args = build_parser().parse_args(
        ["classify", "series.csv", "--plot-evolution", "evidence.gif"]
    )
    assert args.plot_evolution == Path("evidence.gif")


def test_epoch_evidence_maps_have_an_explicit_output_directory():
    args = build_parser().parse_args(
        ["classify", "series.csv", "--plot-epochs", "output/timeseries"]
    )
    assert args.plot_epochs == Path("output/timeseries")


def test_text_summary_has_an_explicit_output():
    args = build_parser().parse_args(
        ["classify", "series.csv", "--output-text", "output/output.txt"]
    )
    assert args.output_text == Path("output/output.txt")


def test_confidence_plot_has_an_explicit_output():
    args = build_parser().parse_args(
        ["classify", "series.npz", "--plot-confidence", "confidence.png"]
    )
    assert args.plot_confidence == Path("confidence.png")


@pytest.mark.model
def test_default_summary_is_concise(capsys):
    inputs = [
        str(path)
        for path in sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    ]
    assert main(["classify", *inputs, "--top-k", "3"]) == 0
    output = capsys.readouterr().out
    assert "Class      Ia" in output
    assert "Two possible redshifts" in output   # quality label is always shown
    assert "68% [" in output and "Top 3" in output
    # the gold cut and the tail detail are verbose-only
    assert "gold-Ia" not in output
    assert "raw p_Ia" not in output


@pytest.mark.model
def test_verbose_summary_shows_gold_cut_and_ambiguous_tail(capsys):
    inputs = [
        str(path)
        for path in sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    ]
    assert main(["classify", *inputs, "--top-k", "3", "--verbose"]) == 0
    output = capsys.readouterr().out
    assert "not in the gold-Ia sample" in output
    assert "5% chance the true redshift is above" in output


@pytest.mark.model
def test_clean_redshift_reads_as_reliable(capsys):
    inputs = [
        str(path)
        for path in sorted(Path("examples/hourglass2_Ia_2791003").glob("spectrum_*.csv"))
    ]
    assert main(["classify", *inputs, "--verbose"]) == 0
    output = capsys.readouterr().out
    assert "Reliable" in output
    assert "in the gold-Ia cosmology sample" in output


@pytest.mark.model
def test_json_flag_prints_only_json(capsys):
    inputs = [
        str(path)
        for path in sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    ]
    assert main(["classify", *inputs, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["classification"]["class"] == "Ia"
    assert "spectra_only" in result["redshift"]
    assert result["input"]["quality"]["model_wavelength_coverage_fraction"] > 0.99


def test_current_model_package_cli_uses_observer_time(
    tmp_path, monkeypatch, capsys
):
    model_dir = tmp_path / "model-package"
    model_dir.mkdir()
    spectrum = tmp_path / "spectrum.csv"
    spectrum.write_text(
        "wavelength,flux,flux_error\n"
        "7500,1.0,0.5\n"
        "8000,2.0,0.5\n"
    )

    class FakeModel:
        def classify(self, **values):
            assert values["observer_time"].tolist() == [62000.0]
            return {
                "format_version": "strider-inference-result-v1",
                "classification": {
                    "class": "Ia",
                    "confidence": 0.8,
                    "probability_type": "calibrated",
                    "probabilities": {"Ia": 0.8, "other": 0.2},
                },
                "redshift": {
                    "z_STRIDER": 0.7,
                    "primary_basin": {"lower_68": 0.65, "upper_68": 0.75},
                    "candidate_basins": [],
                },
                "signal": {
                    "source_probability": 0.9,
                    "grade": "high",
                    "raw_evidence_score": 0.85,
                },
                "input": {"visit_count": 1, "observer_days": [0.0]},
            }

    monkeypatch.setattr("strider.cli.load_model", lambda *_args, **_kwargs: FakeModel())

    assert main(
        [
            "classify",
            str(spectrum),
            "--model",
            str(model_dir),
            "--time",
            "62000",
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["classification"]["class"] == "Ia"
    assert result["redshift"]["z_STRIDER"] == 0.7
