Running STRIDER
===============

Classify one transient with ``model.classify``, or many with ``model.classify_many``. On the command line, use
``strider classify``. The `tutorial <https://github.com/Roman-Supernova-PIT/strider/blob/main/docs/tutorial.ipynb>`_
shows each step, and :doc:`results` explains what comes back.

Inputs
------

A visit is one prism spectrum. For each pixel of each visit, STRIDER needs:

- ``mjd``: the date of the visit, in days;
- ``wavelength``: the observer-frame wavelength, in Å;
- ``flux``: the flux per unit wavelength (f_λ), in any units;
- ``error``: the 1σ error of the flux, in the same units;
- ``mask`` (optional): True for pixels to ignore.

Visits can come in any order. With one visit, any MJD will do. STRIDER ignores pixels whose flux or error is not
finite, or whose error is not positive. A visit should cover the model's wavelength range, given on its model card.
One up to about 130 Å short at the blue end or 320 Å at the red end is still used, and the part it misses counts as
not measured. One further short is left out, and the result says why.

One transient
-------------

A CSV file with one row per pixel is the simplest. The rows with one MJD are one visit.

.. code-block:: python

    import strider
    from strider.io import read_transient

    model = strider.load_model()
    result = model.classify(read_transient("my_transient.csv"))
    print(result)
    result.plot("my_transient.png")

Other inputs work the same way:

- An ECSV file. If its columns have units, STRIDER converts them. This needs astropy.
- A pandas or astropy table: ``model.classify(table)``.
- Lists with one array per visit: ``model.classify(wavelength, flux, error, mjd)``.
- Files from the Roman SMDC, one per visit. They do not hold the date, so give it:
  ``strider.io.read_smdc(["visit1.h5", "visit2.h5"], mjd=[61000.1, 61010.3])``. This needs h5py.

In a CSV, ECSV, pandas or astropy table of one transient, ``time`` can stand for ``mjd``, and ``flux_err`` or
``fluxerr`` for ``error``. A parquet table of spectra needs the exact names.

Many transients
---------------

For many transients, use a parquet table of spectra with one row per visit. Its columns are ``object_id``, ``mjd``,
``wavelength``, ``flux``, ``error`` and, optionally, ``mask``, ``host_redshift`` and ``host_redshift_error``. Each row
of ``wavelength``, ``flux``, ``error`` and ``mask`` is a list. Write such a table with ``strider.io.write_spectra``:

.. code-block:: python

    from strider import Transient, Visit
    from strider.io import read_spectra, write_spectra

    visits = [Visit(wavelength, flux, error, mjd) for wavelength, flux, error, mjd in my_visits]
    write_spectra({"my-transient": Transient(visits)}, "spectra.parquet")
    results = model.classify_many(read_spectra("spectra.parquet"), "results.parquet", plots="figures")

The results file's ending sets its format. Use ``.parquet`` for many transients and long runs, ``.csv`` for a quick
summary, and ``.json`` for detailed results with the redshift posteriors (:doc:`results`). ``plots`` saves each transient's figure in
a folder. If one transient fails, its result is the reason, and the batch carries on. ``strider classify`` exits with
an error only if no transient could be classified. For a long batch written to files, ``keep=False`` keeps only the
failures in memory; the command line does this when it writes a results file.

What you know about a transient
-------------------------------

You can tell STRIDER what you know. Each option does one thing:

- **A redshift range rules out values outside it**: ``redshift_range=(0.5, 1.5)`` (``--zmin 0.5 --zmax 1.5``).
- **A host redshift guides the result, and STRIDER allows for a wrong host**:
  ``model.classify(transient, host_redshift=0.82, host_redshift_error=0.01)``. In a file, use the columns
  ``host_redshift`` and ``host_redshift_error``, the same on every row. STRIDER allows a 5% chance that it is wrong
  (the wrong galaxy, or a failed photometric redshift), so strong data can overrule it. For a spectroscopic redshift,
  ``host_redshift_outlier=0.03`` (``--host-redshift-outlier 0.03``) suits.
- **A redshift density is used as given**: ``redshift_prior=density``, one value per redshift of
  ``model.redshift_grid``, such as a photometric redshift's P(z). It has no allowance for being wrong.
- **A list of classes rules out the others**: ``classes=["Ia", "Iax", "91bg"]`` (``--classes Ia,Iax,91bg``). These
  classes share the probability.
- **Class priors say how common the allowed classes are**: ``class_prior={"Ia": 0.6, "IIP": 0.4}``
  (``--class-prior Ia=0.6,IIP=0.4``). Give a value for each class allowed, so use it with ``classes=["Ia", "IIP"]``,
  or give all classes.

By default, STRIDER assumes the mix of transients Roman is expected to find, given on the model card. In
``classify_many``, the other options apply to every transient; store each transient's host redshift and its error
with its visits.

As the spectra arrive
---------------------

``model.plot_accumulation`` shows how the result builds up as more spectra come in. It classifies the first spectrum,
then the first two, and so on, in time order, and draws one column for each:

- the combined spectrum of those spectra, with its S/N;
- SN Ia and all other classes at every redshift, relative to the most probable;
- the redshift curve of all classes in grey, split into SN Ia and all other classes, with P(Ia) and the main
  redshift.

.. code-block:: python

    model.plot_accumulation(read_transient("my_transient.csv"), path="arrival.png")

These are the first spectra STRIDER can use. A spectrum it leaves out is skipped, so "1 spectrum" may be the second
one observed. Each column is one ``model.classify``, so ``steps=range(1, n + 1)`` is slow for many spectra.

By default the columns use 1, 2, 3, 5, 10 and 20 spectra, those fewer than all of them, then all; with more than 20, 2
is left out to keep six columns. ``steps=range(3, 9)`` uses 3 to 8 spectra instead. ``rows="all"`` shows each class on
its own row. For a simulated transient, ``true_redshift=1.08`` marks its true redshift. It takes the same inputs and
priors as ``model.classify``.

``model.classify_first`` gives the results themselves, without the figure: a dictionary of results by the number of
spectra, so you can follow P(Ia) and the redshift as the spectra arrive:

.. code-block:: python

    for count, result in model.classify_first(transient, steps=range(1, 6)).items():
        print(count, result.p_ia, result.redshift)

Errors that understate the noise
--------------------------------

STRIDER trusts the errors you give. If the spectra are noisier than their errors say, the results are too confident.
``error_scale`` scales the errors up:

- ``error_scale=1.25`` multiplies every error by 1.25 (``--error-scale 1.25``).
- ``error_scale="auto"`` measures the factor from the scatter between neighbouring pixels (``--error-scale auto``).
  It is never below 1, so it only scales errors up. It reads low if the noise is correlated between neighbouring
  pixels, and returns 1 if there is too little faint data to measure, as on a spectrum that is bright everywhere.

Each result gives the factor used, as ``error_scale``. ``classify_many`` measures it for each transient.

Running faster
--------------

On a machine with several cores, ``workers=4`` classifies four transients at once, each on its own core, which is
the fastest way through a large batch (``--workers 4`` on the command line). The results are the same up to rounding, in
the same order. Each worker loads its own copy of the model. In a script, call ``classify_many`` under
``if __name__ == "__main__":``, as Python needs for processes.

On a GPU, load the model with ``device="cuda"``; ``classify_many`` then scores 8 transients together (``batch=``
changes it; ``--device cuda --batch 16``). Each needs about 1 GB of GPU memory at about 20 visits. A batch that runs
out of memory is halved and tried again. Results match a CPU's up to rounding. Apple GPUs (``mps``) do not work. A GPU
only makes large batches faster.

The command line
----------------

- ``strider classify my_transient.csv``: classify one transient, and print its card.
- ``strider classify spectra.parquet -o results.parquet``: classify many, and write the results. ``-o`` can also
  end in ``.csv`` or ``.json``.
- ``strider models``: list the released models. ``--card NAME`` prints a model's card.
- ``strider check``: check the install and a model.

Options of ``strider classify``:

- **Outputs**: ``-o`` (the results file); ``--plot`` (each transient's figure, as ``<name>.png``; ``--plot figures``
  puts them in the folder ``figures``; needs matplotlib); ``--evidence-maps evidence.h5`` and ``--redshift-curves
  redshifts.hdf5`` (:doc:`results`).
- **What you know**: ``--classes``, ``--class-prior``, ``--zmin``, ``--zmax`` and ``--host-redshift-outlier``;
  ``--error-scale 1.25`` or ``--error-scale auto``.
- **Running**: ``--model``, ``--workers``, ``--device`` and ``--batch``. ``--model`` takes a released model's name,
  ``test`` for the bundled test model, or the path to a model package. Without it, STRIDER uses the default model.
