"""Runs written to parquet: the run's record, a stop and a resume, a part killed while written, jobs and merge."""

import json
from importlib.resources import files

import pytest

torch = pytest.importorskip("torch")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import strider  # noqa: E402
from strider.io import Transient, read_spectra, results_schema  # noqa: E402
from strider.runs import Parts, in_job, merge, read_record, run_record  # noqa: E402
from strider.tests.synthetic_package import ARCHIVE  # noqa: E402

EXAMPLE = files("strider") / "data" / "example.parquet"


@pytest.fixture(scope="module")
def model():
    return strider.load_model(ARCHIVE)


@pytest.fixture
def objects():
    return read_spectra(EXAMPLE)


def stopped_after(model, monkeypatch, count):
    """Make ``model``'s batches stop, as a scheduler's stop would, after ``count`` objects."""
    each = model._each

    def stopping(*args):
        for n, found in enumerate(each(*args)):
            if n == count:
                raise KeyboardInterrupt
            yield found

    monkeypatch.setattr(model, "_each", stopping)


def test_a_finished_run_records_how_it_was_run(model, objects, tmp_path):
    source = {"path": "spectra.parquet", "sha256": "ab" * 32, "columns": {"object_id": "SNID"}}
    model.classify_many(objects, tmp_path / "out.parquet", classes=["Ia", "IIP"], source=source)
    record = read_record(tmp_path / "out.parquet")
    assert record["status"] == "complete" and record["format"] == "strider-run-1" and record["job"] is None
    assert record["settings"]["classes"] == ["Ia", "IIP"] and record["settings"]["columns"] == {"object_id": "SNID"}
    assert record["settings"]["fingerprint"] == model.fingerprint and record["input"]["sha256"] == "ab" * 32
    environment = record["sessions"][0]["environment"]
    assert set(environment) == {"python", "numpy", "scipy", "torch", "workers", "threads", "platform"}
    column = pq.read_table(tmp_path / "out.parquet").column("settings_fingerprint").to_pylist()
    assert set(column) == {record["settings_fingerprint"]}
    other = run_record(model, {"classes": ["Ia", "IIP"]}, source=source | {"path": "moved.parquet"})
    assert other["settings_fingerprint"] == record["settings_fingerprint"]  # where the input lies is no setting
    assert run_record(model, {"classes": ["Ia"]})["settings_fingerprint"] != record["settings_fingerprint"]


def test_a_part_left_half_written_is_ignored_and_the_run_resumes(model, objects, tmp_path, monkeypatch):
    output = tmp_path / "out.parquet"
    stopped_after(model, monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, output, chunk=1)
    monkeypatch.undo()
    parts = tmp_path / "out.parquet.parts"
    assert json.loads((parts / "_record.json").read_text())["status"] == "incomplete"
    (parts / "_part-000001.parquet.tmp").write_bytes(b"killed while it was written")
    assert pq.read_table(parts).num_rows == 1  # the folder reads as the results saved so far
    model.classify_many(objects, output, resume=True)
    assert pq.read_table(output).column("object_id").to_pylist() == list(objects)
    assert read_record(output)["status"] == "complete" and not parts.exists()
    with pytest.raises(ValueError, match="exists already .its run is complete."):
        model.classify_many(objects, output, resume=True)


def test_a_resume_needs_the_same_job_settings_and_input(model, objects, tmp_path, monkeypatch):
    output = tmp_path / "out.parquet"
    source = {"path": "spectra.parquet", "sha256": "ab" * 32}
    stopped_after(model, monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, output, chunk=1, source=source)
    monkeypatch.undo()
    for changed, message in (
        ({"job": "0/2"}, "the job changed since it stopped"),
        ({"classes": ["Ia"]}, "the settings changed since it stopped"),
        ({"source": source | {"sha256": "cd" * 32}}, "the input file changed since the run stopped"),
    ):
        with pytest.raises(ValueError, match=message):
            model.classify_many(objects, output, resume=True, **({"source": source} | changed))
    first, transient = next(iter(objects.items()))
    changed = objects | {first: Transient(transient.visits[1:])}
    with pytest.raises(ValueError, match=f"object {first} differs from the one the stopped run saved"):
        model.classify_many(changed, output, resume=True, source=source)
    fewer = {key: value for key, value in objects.items() if key != first}
    with pytest.raises(ValueError, match="holds results for 1 objects that are not in this input"):
        model.classify_many(fewer, output, resume=True, source=source)
    with pytest.raises(ValueError, match="holds results for 3 objects that are not in this input"):
        model.classify_many({}, output, resume=True, source=source)  # the attempt above saved the other two
    model.classify_many(objects, output, resume=True, source=source)  # after all that, the right input still resumes
    assert pq.read_table(output).column("object_id").to_pylist() == list(objects)


def test_two_runs_cannot_write_one_output_at_once(model, tmp_path):
    output, schema, record = tmp_path / "out.parquet", results_schema(model), run_record(model)
    first = Parts(output, schema, record)
    with pytest.raises(ValueError, match="another run is writing"):
        Parts(output, schema, record, resume=True)
    first.close()
    Parts(output, schema, record, resume=True).close()  # once the first lets go


def test_jobs_split_the_input_by_id_and_merge_joins_them(model, objects, tmp_path):
    source = {"path": "spectra.parquet", "sha256": "ab" * 32}
    for k in (0, 1):
        model.classify_many(objects, tmp_path / f"job{k}.parquet", job=(k, 2), source=source)
    jobs = [pq.read_table(tmp_path / f"job{k}.parquet").column("object_id").to_pylist() for k in (0, 1)]
    assert sorted(jobs[0] + jobs[1]) == sorted(objects)
    assert all(in_job(key, (k, 2)) for k in (0, 1) for key in jobs[k])
    record = merge([tmp_path / "job1.parquet", tmp_path / "job0.parquet"], tmp_path / "all.parquet")
    assert record["status"] == "complete" and record["jobs"] == [[0, 2], [1, 2]]
    assert sorted(pq.read_table(tmp_path / "all.parquet").column("object_id").to_pylist()) == sorted(objects)
    with pytest.raises(ValueError, match="are not each of one run's once"):
        merge([tmp_path / "job0.parquet"], tmp_path / "half.parquet")
    assert merge([tmp_path / "job0.parquet"], tmp_path / "half.parquet", partial=True)["status"] == "partial"
    model.classify_many(objects, tmp_path / "other.parquet", job=(1, 2), classes=["Ia"], source=source)
    with pytest.raises(ValueError, match="was run with other settings or another model"):
        merge([tmp_path / "job0.parquet", tmp_path / "other.parquet"], tmp_path / "mixed.parquet")
    model.classify_many(objects, tmp_path / "elsewhere.parquet", job=(1, 2), source=source | {"sha256": "cd" * 32})
    with pytest.raises(ValueError, match="was run on another input file, or other objects of it"):
        merge([tmp_path / "job0.parquet", tmp_path / "elsewhere.parquet"], tmp_path / "mixed.parquet")
    model.classify_many(objects, tmp_path / "chosen.parquet", job=(1, 2), source=source | {"only": ["sn_ia"]})
    with pytest.raises(ValueError, match="was run on another input file, or other objects of it"):
        merge([tmp_path / "job0.parquet", tmp_path / "chosen.parquet"], tmp_path / "mixed.parquet")
    for k in (0, 1):  # jobs of an input given in Python, with no file recorded, merge only as partial
        model.classify_many(objects, tmp_path / f"unknown{k}.parquet", job=(k, 2))
    unknown = [tmp_path / f"unknown{k}.parquet" for k in (0, 1)]
    with pytest.raises(ValueError, match="come from a run whose input file is not recorded"):
        merge(unknown, tmp_path / "unknown.parquet")
    assert merge(unknown, tmp_path / "unknown.parquet", partial=True)["status"] == "partial"


def test_merge_keeps_a_repeated_object_once_unless_its_results_differ(model, objects, tmp_path):
    model.classify_many(objects, tmp_path / "one.parquet")
    table = pq.read_table(tmp_path / "one.parquet")
    slower = table.set_column(table.schema.get_field_index("seconds"), "seconds", pa.array([1e6] * table.num_rows))
    pq.write_table(slower, tmp_path / "slower.parquet")  # the same run's metadata, other times
    merge([tmp_path / "one.parquet", tmp_path / "slower.parquet"], tmp_path / "joined.parquet", partial=True)
    joined = pq.read_table(tmp_path / "joined.parquet")
    assert joined.num_rows == len(objects) and max(joined.column("seconds").to_pylist()) < 1e6
    other = table.set_column(table.schema.get_field_index("p_ia"), "p_ia", pa.array([0.5] * table.num_rows))
    pq.write_table(other, tmp_path / "other.parquet")
    with pytest.raises(ValueError, match="has different results in two of the files"):
        merge([tmp_path / "one.parquet", tmp_path / "other.parquet"], tmp_path / "clash.parquet", partial=True)


def test_resume_with_no_run_to_carry_on_starts_one(model, objects, tmp_path):
    model.classify_many(objects, tmp_path / "out.parquet", resume=True)
    assert pq.read_table(tmp_path / "out.parquet").num_rows == len(objects)


def test_a_finished_run_sums_up_what_it_holds_and_says_how_it_goes(model, objects, tmp_path, monkeypatch, caplog):
    import logging

    from strider import model as module

    monkeypatch.setattr(module, "PROGRESS_SECONDS", 0.0)
    with caplog.at_level(logging.DEBUG, logger="strider"):
        model.classify_many(objects | {"bad": "ValueError: a visit has no finite mjd"}, tmp_path / "out.parquet")
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == "classifying on 1 worker"
    assert messages[-2].startswith("4 done, 1 failed, after ") and messages[-1] == "finished"  # the counts once
    assert any(message.startswith("sn_ia: ok in ") and "peak memory" in message for message in messages)
    assert "bad: failed" in messages and any(message.startswith("1 done, 0 failed") for message in messages)
    record = read_record(tmp_path / "out.parquet")
    found = record["summary"]
    assert found["objects"] == 4 and found["status"] == {"failed": 1, "ok": 3}
    assert found["seconds_per_object"]["max"] >= found["seconds_per_object"]["median"] > 0
    assert found["largest_memory_mb"] > 50 and found["cpu_seconds"] > 0
    assert record["started"] <= record["finished"]
    assert sum(found["light_curve_check"].values()) == 3  # the failure has none


def test_info_reports_a_stopped_run_from_its_parts(model, objects, tmp_path, monkeypatch, capsys):
    from strider.cli import main

    stopped_after(model, monkeypatch, 2)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=1)
    main(["info", str(tmp_path / "out.parquet.parts")])
    shown = capsys.readouterr().out
    assert shown.startswith("Incomplete run of") and "Saved so far: 2 objects" in shown


def test_results_are_saved_at_least_every_few_minutes(model, objects, tmp_path, monkeypatch):
    from strider import model as module

    monkeypatch.setattr(module, "SAVE_SECONDS", 0.0)
    stopped_after(model, monkeypatch, 2)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=100)
    assert len(list((tmp_path / "out.parquet.parts").glob("part-*.parquet"))) == 2  # one a result, not at the end


def test_each_session_of_a_resumed_run_is_recorded(model, objects, tmp_path, monkeypatch):
    stopped_after(model, monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=1)
    monkeypatch.undo()
    model.classify_many(objects, tmp_path / "out.parquet", resume=True, workers=2)
    sessions = read_record(tmp_path / "out.parquet")["sessions"]
    assert len(sessions) == 2 and sessions[1]["environment"]["workers"] == 2
    assert sessions[1]["environment"]["threads"] == 1  # each worker runs on one thread


def test_a_disk_that_fails_to_flush_stops_the_run_and_a_platform_without_locks_refuses_it(model, tmp_path, monkeypatch):
    import errno
    import os
    import sys

    from strider import runs

    def failing(descriptor):
        raise OSError(errno.EIO, "input/output error (faked)")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", failing)
        with pytest.raises(OSError, match="input/output error"):
            Parts(tmp_path / "out.parquet", results_schema(model), run_record(model))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "fcntl", None)
        with pytest.raises(ValueError, match="needs file locks, which this platform lacks"):
            runs._locked(tmp_path / "_lock")


def test_transients_scored_together_have_no_time_of_their_own_in_the_summary(model, objects, tmp_path):
    model.classify_many(objects, tmp_path / "out.parquet", batch=3)
    found = read_record(tmp_path / "out.parquet")["summary"]
    assert found["seconds_per_object"] == {"measured": 0, "total": None, "median": None, "p90": None, "max": None}
    assert found["cpu_seconds"] is None and found["objects"] == 3


def test_a_stop_with_workers_still_saves_the_results_already_finished(model, objects, tmp_path):
    import time

    def then_a_broken_table():
        yield from list(objects.items())[:2]
        time.sleep(20)  # time for the two workers to start and finish them
        raise ValueError("the rows of object x are not together (faked)")

    with pytest.raises(ValueError, match="not together"):
        model.classify_many(then_a_broken_table(), tmp_path / "out.parquet", workers=2)
    assert sorted(pq.read_table(tmp_path / "out.parquet.parts").column("object_id").to_pylist()) == sorted(
        list(objects)[:2]
    )


def test_waiting_on_workers_wakes_to_save_and_hands_over_results_before_a_dead_worker():
    from concurrent.futures import Future

    from strider.model import _finished_jobs

    waiting = Future()
    assert list(_finished_jobs({waiting: "slow"}, timeout=0.01)) == [(None, None)]
    worked, died = Future(), Future()
    worked.set_result("its outcome")
    died.set_exception(RuntimeError("a worker died (faked)"))
    found = []
    with pytest.raises(RuntimeError, match="a worker died"):
        for item in _finished_jobs({died: "lost", worked: "kept"}):
            found.append(item)
    assert found == [("kept", "its outcome")]


def test_a_merge_holds_its_output_as_a_run_does_and_a_resume_keeps_its_input_checksum(
    model, objects, tmp_path, monkeypatch
):
    from strider.runs import RunBusy

    source = {"path": "spectra.parquet", "sha256": "ab" * 32}
    model.classify_many(objects, tmp_path / "all.parquet", source=source)
    held = Parts(tmp_path / "merged.parquet", results_schema(model), run_record(model))
    with pytest.raises(RunBusy, match="another run or merge is writing"):
        merge([tmp_path / "all.parquet"], tmp_path / "merged.parquet")
    held.close()
    stopped_after(model, monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=1, source=source)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="recorded its input file, and this run gives none"):
        model.classify_many(objects, tmp_path / "out.parquet", resume=True)


def test_a_folder_that_cannot_be_locked_is_not_left_behind(model, tmp_path, monkeypatch):
    import errno
    import fcntl

    def refused(*args):
        raise OSError(errno.ENOLCK, "no locks available (faked)")

    monkeypatch.setattr(fcntl, "flock", refused)
    with pytest.raises(ValueError, match="cannot lock"):
        Parts(tmp_path / "out.parquet", results_schema(model), run_record(model))
    assert not (tmp_path / "out.parquet.parts").exists()


def test_a_stop_while_a_result_is_handled_still_saves_what_workers_finished(model, objects, tmp_path, monkeypatch):
    import time

    from strider import model as module

    calls = []

    def slow_then_stopped(self, key, outcome):  # the first result's handling takes long, then a stop comes
        calls.append(key)
        if len(calls) == 1:
            time.sleep(20)  # the other two workers' jobs finish meanwhile
            raise KeyboardInterrupt

    monkeypatch.setattr(module._Progress, "add", slow_then_stopped)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", workers=2)
    saved = pq.read_table(tmp_path / "out.parquet.parts").column("object_id").to_pylist()
    assert sorted(saved) == sorted(objects) and len(calls) == 1  # the hand-over only keeps the results


def test_numpy_whole_numbers_are_taken_as_plain_ones(model, objects, tmp_path):
    import numpy as np

    model.classify_many(objects, tmp_path / "out.parquet", max_spectra=np.int64(2), workers=np.int64(1))
    assert read_record(tmp_path / "out.parquet")["settings"]["max_spectra"] == 2


def test_a_merge_refused_because_another_run_took_its_folder_leaves_that_folder(model, objects, tmp_path, monkeypatch):
    from strider import runs
    from strider.runs import RunBusy

    model.classify_many(objects, tmp_path / "all.parquet", source={"path": "s.parquet", "sha256": "ab" * 32})

    def taken(path):  # a resumed run locks the folder between the merge making it and locking it
        (path.parent / "_record.json").write_text("{}")
        raise RunBusy(f"another run is writing {path.parent}")

    monkeypatch.setattr(runs, "_locked", taken)
    with pytest.raises(RunBusy):
        merge([tmp_path / "all.parquet"], tmp_path / "merged.parquet")
    assert (tmp_path / "merged.parquet.parts" / "_record.json").exists()


def test_a_stop_while_results_are_saved_waits_until_they_are(model, objects, tmp_path, monkeypatch):
    import os
    import signal

    from strider import runs

    write = runs.Parts.write

    def stopped_while_writing(self, table):
        os.kill(os.getpid(), signal.SIGINT)  # Ctrl-C as the first save begins
        write(self, table)

    monkeypatch.setattr(runs.Parts, "write", stopped_while_writing)
    with pytest.raises(KeyboardInterrupt):
        model.classify_many(objects, tmp_path / "out.parquet", chunk=2)
    assert pq.read_table(tmp_path / "out.parquet.parts").num_rows == 2  # the save went through, then the stop
