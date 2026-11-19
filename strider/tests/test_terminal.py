"""How ``strider classify`` looks: the ruled report, the line that ends a run, the live line, and a stop."""

import io
import logging
import re
import warnings

import pytest

pytest.importorskip("torch")

import strider  # noqa: E402
from strider import cli, terminal  # noqa: E402
from strider.tests.synthetic_package import ARCHIVE  # noqa: E402

PLAIN = re.compile(r"\033\[[0-9;]*m")


class Stream(io.StringIO):
    """A stream that is a terminal or not."""

    def __init__(self, tty=False):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


@pytest.fixture(scope="module")
def result():
    return strider.load_model(ARCHIVE).classify(strider.load_example())


def test_the_report_is_ruled_and_says_what_the_report_says(result):
    report = terminal.report(result, "sn_ia", terminal.Style(Stream()), "figures/sn_ia.png").splitlines()
    assert report[0].startswith("── sn_ia ─") and result.model not in report[0]
    assert len(report[0]) == terminal.WIDTH and report[-2] == "─" * terminal.WIDTH
    assert report[-1] == "   Figure    figures/sn_ia.png"
    plain = result.report("sn_ia").splitlines()
    value, interval = plain[2][10:].split(", ", 1)
    assert report[1:4] == ["   " + plain[1], "   Redshift  " + value, "   " + " " * 10 + interval]


def test_in_a_terminal_the_report_has_colour_unless_no_color_is_set(result, monkeypatch):
    coloured = terminal.report(result, "sn_ia", terminal.Style(Stream(tty=True)))
    strength = terminal.STRENGTH[result.match_strength]
    assert f"\033[{strength};1m{result.match_strength}\033[0m" in coloured and "strength" not in coloured
    top, p = result._ranked()[0]
    assert f"\033[{terminal.TIER[strider.io._tier(p)]};1m{top} " in coloured  # gold, silver, bronze or red
    assert PLAIN.sub("", coloured) == terminal.report(result, "sn_ia", terminal.Style(Stream()))
    monkeypatch.setenv("NO_COLOR", "1")
    assert terminal.report(result, "sn_ia", terminal.Style(Stream(tty=True))) == PLAIN.sub("", coloured)


def test_the_last_line_says_how_many_how_long_and_where():
    style = terminal.Style(Stream())
    assert terminal.finished(style, 1, 0, 79) == "✓ 1 transient classified in 1 min 19 s"
    assert terminal.finished(style, 2, 1, 5) == "✓ 2 transients classified, 1 failed, in 5 s"
    assert terminal.finished(style, 412, 3, 7500, "results.parquet") == (
        "✓ 412 classified, 3 failed, in 2 h 05 min → results.parquet"
    )


def test_the_live_line_follows_the_log_and_puts_warnings_above_it(capsys):
    log = logging.getLogger("strider")
    before = log.level, log.propagate, warnings.showwarning
    with terminal.LiveLine(name="sn_ia") as live:
        assert live.text() == "Classifying sn_ia   0 s"
        log.debug("sn_ia: ok in 3 s", extra={"done": 1, "failed": 0})
        assert live.text().startswith("1 done   ")
        log.info("1 done, 0 failed, after 3 s", extra={"done": 2, "failed": 1})
        assert live.text().startswith("2 done   1 failed   ")
        warnings.warn("the figure could not be saved", stacklevel=1)
    err = capsys.readouterr().err
    assert "warning: the figure could not be saved\n" in err and "sn_ia: ok" not in err  # each transient with verbose
    assert (log.level, log.propagate, warnings.showwarning) == before
    assert terminal.LiveLine(workers=4).text() == "Classifying transients on 4 workers   0 s"
    counted = terminal.LiveLine(workers=4, total=10)
    assert counted.text() == "Classifying 10 transients on 4 workers   0 s"
    counted.started -= 32
    counted.emit(logging.makeLogRecord({"levelno": logging.INFO, "msg": "", "done": 2, "failed": 0}))
    assert counted.text() == "2 of 10 done   32 s   16 s each   2 min 08 s left"
    counted.started, counted.last = counted.started - 10, counted.last - 10  # 10 s on, nothing new finished
    assert counted.text() == "2 of 10 done   42 s   16 s each   1 min 58 s left"  # down, not up
    counted.started, counted.last = counted.started - 200, counted.last - 200  # past the estimate
    assert counted.text().endswith("16 s each   finishing")


def test_a_run_counts_its_transients_from_the_ids_alone(tmp_path):
    from argparse import Namespace

    from strider.io import read_spectra
    from strider.runs import in_job

    table = cli.EXAMPLE.parent / "example.parquet"
    run = Namespace(columns=None, only=None, resume=False)
    assert cli.planned(table, run, None, None) == 3
    assert cli.planned(table, Namespace(columns=None, only=["sn_ia"], resume=False), None, None) == 1
    keys = list(read_spectra(table))
    assert cli.planned(table, run, (0, 2), None) == sum(in_job(key, (0, 2)) for key in keys)
    renamed = cli.EXAMPLE.parent / "example_own_names.parquet"
    assert cli.planned(renamed, Namespace(columns={"object_id": "SNID"}, only=None, resume=False), None, None) == 3
    assert cli.planned(tmp_path / "one.csv", run, None, None) == 1


def test_a_stop_says_so_and_how_to_carry_on(monkeypatch, tmp_path, capsys):
    def stopped(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "classify", stopped)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["classify", "spectra.parquet", "-o", str(tmp_path / "results.parquet")])
    assert excinfo.value.code == 130
    assert capsys.readouterr().err == "strider: stopped; carry on with --resume\n"


def test_the_count_reads_the_ids_a_batch_at_a_time(monkeypatch):
    from argparse import Namespace

    import pyarrow.parquet as pq

    def whole(*args, **kwargs):
        raise AssertionError("the whole id column was read at once")

    monkeypatch.setattr(pq, "read_table", whole)  # a table of pixels has many rows for each transient
    by_pixel = cli.EXAMPLE.parent / "example_by_pixel.parquet"
    assert cli.planned(by_pixel, Namespace(columns=None, only=None, resume=False), None, None) == 3


def test_a_resumed_run_counts_this_session_and_says_what_the_file_holds():
    style = terminal.Style(Stream())
    assert terminal.finished(style, 1, 0, 2, "results.parquet", 10000) == (
        "✓ 1 classified in 2 s → results.parquet (10000 in the file)"
    )
    assert terminal.finished(style, 3, 0, 2, "results.parquet", 3).endswith("→ results.parquet")


def test_a_stopped_merge_does_not_suggest_resume(monkeypatch, tmp_path, capsys):
    def stopped(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "merge", stopped)
    with pytest.raises(SystemExit):
        cli.main(["merge", "a.parquet", "b.parquet", "-o", str(tmp_path / "results.parquet")])
    assert capsys.readouterr().err == "strider: stopped\n"


def test_a_report_names_only_a_figure_this_run_saved(tmp_path):
    import os

    assert cli.saved_figure(None, "sn_ia", 0.0) is None
    assert cli.saved_figure(tmp_path, "sn_ia", 0.0) is None  # never saved
    old = tmp_path / "sn_ia.png"
    old.write_bytes(b"png")
    os.utime(old, (1e9, 1e9))
    assert cli.saved_figure(tmp_path, "sn_ia", 2e9) is None  # from an earlier run
    assert cli.saved_figure(tmp_path, "sn_ia", 1e9) == old
