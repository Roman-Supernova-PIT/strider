When something goes wrong
=========================

What a message means
--------------------

- **"the rows of object ... are not together"**: ``strider classify`` reads a parquet table one transient at a time,
  so each transient's rows must be next to each other. Sort the table by ``object_id``.
- **"... is missing the column(s) ..."** or **"... has no column for ..."**: give the table's own column names with
  ``--columns`` (:doc:`usage`).
- **"... exists already"**: STRIDER never overwrites a parquet results file. Give a new name, or remove the old file.
- **"... is left from a run that stopped"**: carry it on with ``--resume``, or remove the ``.parts`` folder to start
  again (:doc:`running_at_scale`).
- **"cannot resume ...: ... changed since it stopped"**: the input, model, settings, STRIDER version, job or ``--only``
  list are not the stopped run's. Write to a new file.
- **"another run is writing ..."**: two jobs were given the same results file. Give each its own.
- **"cannot lock ..."**: the folder is on a file system without file locks. Write the results somewhere else.
- **A transient's result is a reason, not a class**: the reason says what was wrong with its input, such as a visit
  with no date, or one that misses the model's wavelengths.
- **``status`` is ``spectra_only``**: the brightness comparison could not be worked out for that transient, so its
  result uses the spectra alone. ``brightness_failure`` says why.

Telling us about a problem
--------------------------

Open an issue at https://github.com/Roman-Supernova-PIT/strider/issues with:

1. what ``strider info`` prints, and, for a run, what ``strider info results.parquet`` prints;
2. the command you ran, and what it printed;
3. a saved transient that shows the problem.

A saved transient is a folder with its input, as STRIDER read it, the run's settings and the result:

.. code-block:: console

    strider classify spectra.parquet --only SNID123 --save-transient saved_transients [the other options of your run]

``strider rerun saved_transients`` classifies each saved transient again, with the model and settings it records, and
says whether the result is the same. If not, it says which numbers changed, and how the versions and machine differ
from the run's. Zip the folder and attach it to the issue.
