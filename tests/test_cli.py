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


def test_evidence_evolution_has_an_explicit_output():
    args = build_parser().parse_args(
        ["classify", "series.csv", "--plot-evolution", "evidence.gif"]
    )
    assert args.plot_evolution == Path("evidence.gif")


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
