Reading the results
===================

``print(result)`` gives its card:

.. code-block:: text

    STRIDER · transient                                         strider-15class-1.0
    Class     Ia 100%
    Redshift  0.869 (100% within 0.1) · 68% range 0.864–0.873
    Match     high strength
    Data      11 visits over 100 days · S/N 4.8

It reads:

- **Class**: the most likely classes, up to three.
- **Redshift**: the most likely redshift, the probability that the true one is within 0.1 of it, and the central 68%
  range.
- **Match**: how strong the best match is (high, medium or low) and any warnings.
- **Data**: the visits used, the days they span and their S/N, the median signal-to-noise per pixel of the visits
  combined.
- **Checks**, only when one applies: a weak source, few classes trained at the redshift, untrained subtypes, or a
  brightness comparison that failed.

``result.plot()`` draws the figure: the spectra (one from each visit) and their combination, with the four bands
shaded and the rest wavelength at the main redshift on top; the combined spectrum against the best-matching reference
of the two most likely classes, each at its own redshift; each visit's brightness in the four bands, against the SN
Ia light curves fitted to it at the most probable SN Ia redshift, in days from the fitted peak; every class at every
redshift, with the main and second redshift peaks circled; and the class and redshift probabilities. In the
brightness panel, each band's steady level, such as host light, is taken off the points, and the shading shows how
far SN Ia light curves stray from the fit. The redshift curve of all classes is grey, and it is split into SN Ia and
all other classes, so you can see whether each peak comes from SN Ia or from another class. Other includes 91bg and
Iax. It needs matplotlib.

The results use a prior: how common each class is at each redshift among the transients Roman will find. They also
use your redshift prior and class priors, if you give them.

Which numbers to use
--------------------

- Check the flags first. A raised flag means the result needs care.
- To select SN Ia, cut on ``p_ia``. A higher cut-off gives a purer but smaller sample.
- For the class, use ``top_class`` with ``top_probability``. A low ``top_probability`` means the class is uncertain.
- ``match_strength`` and ``warnings`` say how clearly the data pick one class, whatever the class mix.
- For the redshift, use ``redshift`` and ``redshift_68``. With several redshift peaks, ``redshift_peaks`` lists them.
  If you treat the transient as a SN Ia, use ``result.redshift_given("Ia")`` instead: it is the redshift if the
  transient is a SN Ia.

There are three kinds of redshift result:

- ``redshift`` and ``redshift_posterior`` combine all classes.
- ``result.redshift_given(name)`` assumes the transient is of that class.
- ``class_redshift_posterior`` holds one row per class, each the redshift posterior given that class.

The fields
----------

- ``class_probabilities``: the probability of each class.
- ``p_ia``: the probability of a type Ia supernova.
- ``top_class`` and ``top_probability``: the most likely class and its probability. ``second_class`` and
  ``second_probability``: the next one.
- ``equal_prior_probabilities``: the class probabilities if every class were equally common.
- ``redshift``: the most likely redshift: the peak of the redshift posterior where the most probability lies within
  0.1 of it.
- ``redshift_probability``: the probability that the true redshift is within 0.1 of ``redshift``.
- ``redshift_peaks``: each likely redshift as (redshift, probability within 0.1), the main one first, then by
  probability. A second peak is listed if it holds at least 10% within 0.1 and is clearly separate: between it and
  each higher peak, the probability drops below half its own.
- ``redshift_68`` and ``redshift_95``: the central 68% and 95% intervals of the redshift.
- ``redshift_posterior``: the probability that the redshift is in each cell. It sums to 1.
- ``redshift_grid`` and ``redshift_cell_widths``: the centre and width of each cell. Divide ``redshift_posterior``
  by the widths to get a density.
- ``class_redshift_posterior``: the redshift posterior given each class, one row per class. Each row sums to 1, also
  for a class you leave out with ``classes``, whose probability is 0. A row is NaN for a class with no posterior at
  the redshifts allowed: none where it had training transients, or none inside ``redshift_range``.
  ``result.redshift_given(name)`` gives its most likely redshift, its 68% and 95% intervals, and the row.
- ``class_redshifts``: the most likely redshift of each class, by the same rule as ``redshift``.
- ``log_evidence``: the evidence map, the evidence for each class at each redshift before any prior. Add the log of
  your own prior probability of each class and redshift cell, and normalise, to get your posterior.
- ``source_probability``: the probability that the visits hold a transient at all.
- ``untrained_share`` and the flag ``untrained_subtypes`` are for models whose classes pool several subtypes; for
  ``strider-15class`` they are always 0 and False.
- ``trained_classes``: the classes the model had training transients of at ``redshift``.
- ``flags``:

  - ``weak_source``: there is probably no transient. ``source_probability`` is below the cut-off on the model card.
  - ``several_redshift_peaks``: the redshift has more than one likely value. Look at ``redshift_posterior``.
  - ``few_classes_trained``: fewer than half the classes had training transients at ``redshift``, so the classes
    that did not cannot be ruled out.
  - ``brightness_failed``: the brightness term could not be computed, so the result uses the spectra alone.

- ``best_match``: the class and redshift that match the data best, before the expected class mix and your priors.
- ``next_best_class``: the best other class, at its own best redshift.
- ``match_gap``: how far the best match is ahead of the next-best class, in log units. It is not a probability.
- ``match_strength``: ``high``, ``medium`` or ``low``, from the match gap at levels fixed with the model: for
  ``strider-15class``, a gap of 5 or more is high, 2 to 5 medium, and under 2 low.
- ``warnings``: why a match may be less clear, as short sentences:

  - ``second redshift near z ...``: another redshift of the best class matches nearly as well.
  - ``spectra and brightness disagree``: the spectra alone and the brightness alone favour different classes or
    redshifts.

- ``visits_used``: the visits used, by their position in your input, counting from 0.
- ``days_observed``: the days from the first visit used to the last.
- ``signal_to_noise``: the median signal-to-noise per pixel of the visits used, combined, with their errors
  multiplied by ``error_scale``.
- ``dropped_visits``: the visits left out, by position, each with the reason.
- ``error_scale``: the factor the errors were multiplied by (:doc:`usage`); 1 if they were used as given.
- ``redshift_range``: the redshift range you restricted the result to, or none. The redshifts and intervals then lie
  inside it.
- ``model`` and ``fingerprint``: the model's name and version, such as ``strider-15class-1.0``, and a code that
  identifies the model: its network, brightness term, priors and settings. ``package_sha256`` (below) identifies
  every file of its package.

The results file
----------------

The file's ending sets its format. Use parquet for many transients and long runs, CSV for a quick summary, and
JSON for detailed results with the redshift posteriors (evidence maps are saved separately):

- ``.parquet``: a table with one row per transient, with the columns below and ``redshift_posterior``.
- ``.csv``: the same rows, without ``redshift_posterior``.
- ``.json``: each transient's full result, by name, with the redshift grid and every redshift posterior.

``strider.io.results_schema(model).names`` lists every column. Besides the fields above that are single values,
there are:

- ``object_id``: the transient's name in your input;
- ``probability_<class>`` and ``equal_prior_<class>``: one column for each class;
- ``redshift_68_lower``, ``redshift_68_upper``, ``redshift_95_lower`` and ``redshift_95_upper``: the interval ends;
- ``redshift_ia``, ``redshift_ia_68_lower`` and so on: the same for the redshift if the transient is a SN Ia;
- ``second_peak_redshift`` and ``second_peak_probability``: the second redshift peak, or empty;
- ``best_match_class`` and ``best_match_redshift``: the best match;
- ``top_class_redshift`` and ``second_class_redshift``: the redshift of each of the two most likely classes;
- ``package_sha256``: a checksum covering every file of the model package;
- ``strider_version``: the version of STRIDER that wrote the row;
- ``failure``: why the transient could not be classified, or empty.

The results do not record the priors or options you gave; keep the command with them.

Each entry of ``dropped_visits`` reads ``"<position>: <reason>"``. In a CSV file, ``visits_used``,
``dropped_visits`` and ``warnings`` are text, the warnings separated by semicolons. The parquet table's metadata holds
``redshift_grid`` and ``redshift_cell_widths``. Tables from runs of one model have the same columns, so you can stack
them.

Two more files are for analysis:

- **Evidence maps** (``strider.io.write_evidence_maps``, an HDF5 file): ``log_evidence`` for every transient, with
  ``object_id``, ``error_scale``, ``classes``, the redshift grid and the model's own prior, ``log_survey_prior``.
  Use them to apply your own mix of classes or redshift information without running STRIDER again. Where the model
  had no training transients, the evidence is -inf. Writing them needs h5py.
- **Redshift curves** (``strider.io.write_redshift_curves``): the redshift posteriors in the format of qp, which LSST
  and DESC redshift tools read, with ``object_id`` and ``redshift``. Writing them needs qp (``pip install
  qp-prob``).

In a batch (``classify_many``, or the command line), the results and the evidence maps are written as the batch
goes, ``chunk`` transients at a time. If the run stops with an error or you stop it (Ctrl-C, or a scheduler's
SIGTERM), it first writes the transients done so far. A run killed outright cannot: a CSV file keeps the rows written
before, and a parquet file's rows so far are in the folder ``<file>.parquet.parts``, which
``pyarrow.parquet.read_table`` reads (remove it before running again). JSON results and the redshift curves are
written only at the end, so for long batches use parquet.
