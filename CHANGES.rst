Changes
=======

0.2.0 (2026-11-19)
------------------

For runs of many transients, on a cluster or your own computer:

- A run written to parquet saves as it goes and, if it stops, carries on with ``--resume``.
- ``--job K/N`` splits a run into jobs, and ``strider merge`` joins their results.
- ``strider info`` describes this install, or the run behind a results file: its model, settings, input, time and
  memory.
- Each results row says how its transient was run: ``status``, ``warning_codes``, the spectra used, and its time, CPU
  time and memory.
- ``--max-spectra N`` uses only the first N spectra of each transient, which bounds the time one with many takes.
- A table of spectra is read one transient at a time, so its size does not matter. It can keep its own column names
  (``--columns``), give wavelengths in nm or microns and flux per unit frequency, and have one row per visit or per
  pixel.
- ``--only`` classifies chosen transients. ``--save-transient`` saves one with its settings and result, and ``strider
  rerun`` classifies it again to check it.

Results:

- The brightness comparison searches peak dates about 2.5 times faster by default. The results are close to 0.1's but
  not the same: a transient near a level of match strength can get the other level. ``--full-scan`` uses 0.1's scan.
- Each result checks the light curve's errors against its scatter (``light_curve_check``).
  ``--light-curve-errors correct`` scales them up.

What you see:

- Figures need nothing extra: ``pip install .`` installs matplotlib and cmasher.
- ``print(result)`` gives the result's report, also ``result.report()``; ``result.card()`` still works.
- In a terminal, ``strider classify`` shows how far it has got on one line, with the time left, and each report in
  colour.
- A new figure: the result in a box at the top, the spectra over their spread, the best match's model spectrum and
  light curves, and open circles for faint brightness points.

0.1.0 (2026-11-13)
------------------

The first public release, with the model ``strider-15class`` 1.0.
