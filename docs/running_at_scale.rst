Running many transients
=======================

For thousands of transients, on a cluster or your own computer, write the results to a parquet file. A run like this
can stop and carry on, be split into several jobs, and keeps a record of how it was made.

A run
-----

.. code-block:: console

    strider classify spectra.parquet -o results.parquet --workers 32

STRIDER reads the table one transient at a time, so its size does not matter. The rows of each transient must be
together. As it goes, STRIDER saves the results in the folder ``results.parquet.parts``: every 100 transients
(``--save-every``) and, with ``--workers``, at least every 5 minutes or so. When every transient is done, the saves
are joined into ``results.parquet`` and the folder is removed.

If the run stops, for any reason, the folder keeps what was saved. Run the same command with ``--resume`` and it
carries on: it skips the transients already saved, and classifies again the ones that were running when it stopped.
``--resume`` needs the same input file, model, settings, STRIDER version, job and ``--only`` list, and stops if any of
them changed. For new or corrected spectra, write to a new file. With nothing saved yet, ``--resume`` simply starts the
run, so a cluster job that may be restarted can use it from the start. A run that has finished is not run again:
``--resume`` then stops and says the file is complete.

STRIDER never overwrites a parquet results file: give a new name, or remove the old file first. Only one run at a time
can write to a file.

``pyarrow.parquet.read_table("results.parquet.parts")`` reads what a stopped run saved.

Splitting a run into jobs
-------------------------

``--job K/N`` makes a run job K of N, with K from 0 to N - 1, so N jobs can share one input. Each transient goes to
one job by its ``object_id``, the same way on any machine. Give each job its own results file, then join them:

.. code-block:: console

    strider classify spectra.parquet -o job0.parquet --job 0/4 --resume
    ...
    strider classify spectra.parquet -o job3.parquet --job 3/4 --resume
    strider merge job0.parquet job1.parquet job2.parquet job3.parquet -o results.parquet

``strider merge`` needs every job, once each, from the same input file, model and settings. ``--partial`` joins only
the jobs given. In Python: ``model.classify_many(..., job=(k, n), resume=True)`` and ``strider.runs.merge``.

The record of a run
-------------------

Each parquet results file keeps a record of its run: the STRIDER version, the model, every setting that changes the
results, the input file and its checksum, the job, the versions and machine used, when it started and finished, and
a summary of what it holds. ``strider info results.parquet`` shows it; add ``--json`` for programs, or use
``strider.runs.read_record`` in Python. ``strider info results.parquet.parts`` shows how far a run has got.

Each row also says how its transient was run:

- ``status``: ``ok``; ``spectra_only`` if the brightness comparison failed, so the spectra alone were used; or
  ``failed``;
- ``warning_codes``;
- ``input_fingerprint``: a code for the transient's input, the same on any machine;
- the number of spectra used and the date of the last one;
- ``settings_fingerprint``: a code for the run's model and settings;
- ``seconds`` and ``cpu_seconds``: how long the transient took;
- ``peak_memory_mb``: the most memory the process had used by then, in MB.

As it runs
----------

In a terminal, ``strider classify`` shows how far it has got on one line that updates in place, and ends with how many
it classified and how long it took. In a job's log, it writes a line about once a minute instead, and one at the end.
``--verbose`` adds a line for each transient, with its time and memory. In Python, these go to the ``strider`` logger.
Set ``NO_COLOR`` to turn off colour.

Time and memory
---------------

Each worker holds the model and one transient: 1 to 2 GB of memory, a little more the longer it runs. The time per
transient depends mostly on its light curve. It ranges from seconds to more than half an hour on one core, and is a
few minutes for a typical transient. ``--max-spectra 16`` keeps every transient to a few minutes, at some cost for
distant, faint transients (:doc:`usage`). ``strider info`` sums up the time and memory a run used, to plan the next.
On a Slurm cluster, ``seff <job id>``, where it is installed, shows the memory and CPU time a finished job used.
