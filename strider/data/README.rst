``example.parquet`` holds the spectra of three simulated transients, used by ``strider.load_example`` and ``strider check``.
``example.csv`` holds the first of them, one row per pixel, for the command-line example.
``example_expected.json`` holds the results ``strider check`` expects for them, for each model and peak-date scan.
``example_by_pixel.parquet`` holds the same three, one row per pixel with each visit's ``visit_id``.
``example_own_names.parquet`` holds them one row per visit under other names, with wavelength in microns and flux per
unit frequency, as another pipeline might give them; read it with
``--columns object_id=SNID,wavelength=lam,error=flux_error --wavelength-unit micron --flux-density f_nu``.
