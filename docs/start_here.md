# One object through STRIDER

Each transient is represented by one or more observer-frame prism spectra,
with measured flux, reported uncertainty and an observation date for each visit.

## Before inference

The fixed reference bank is built once from the training split. Clean simulated
spectra may be placed on a rest-wavelength and broad-phase grid using training
truth during this offline step. The bank records its source split, configuration
digest and edge-weighting semantics.

Training truth is not copied into an inference object. In particular, the model
does not receive the object's true class, true redshift, simulated clean flux or
truth-derived phase.

## Processing the spectra

1. [`deployment.py`](../src/strider/deployment.py) or
   [`data/dataset.py`](../src/strider/data/dataset.py) validates the measured
   spectra, sorts them chronologically and resamples each visit once onto the
   observer-frame grid.
2. Each spectrum is scaled by an uncertainty estimate for numerical stability.
   The uncertainty at each wavelength is retained for weighting measurements.
3. [`model/coadd.py`](../src/strider/model/coadd.py) reverses the visit scaling
   and forms one inverse-variance accumulated spectrum with propagated error.
4. [`model/roman_reference.py`](../src/strider/model/roman_reference.py) keeps
   the accumulated measured flux unchanged, then forms a normalized full
   spectrum and a continuum-removed view.
5. Both views are aligned and compared with the fixed reference bank at every
   candidate redshift. Fine simulation classes are mapped explicitly to the
   configured reporting classes.
6. At most eight temporal spectra are retained. If more are available, STRIDER
   divides the sorted visit indices into eight blocks and selects the visit
   with the highest measured median S/N in each block. These blocks need not
   span equal durations.
7. Time intervals from the first selected visit are divided by
   `1 + candidate redshift`. The model combines comparisons over possible
   starting phases and uses uncertainty-weighted relative brightness. It does
   not receive the simulated starting phase.
8. Spectral and temporal scores form one joint class-redshift surface.
9. [`model/posterior.py`](../src/strider/model/posterior.py) applies the declared
   prior and redshift-cell widths before normalizing the joint distribution.
10. A separate measured-signal component returns a source/noise score, with
    calibrated source probability and grade when fitted. Its interpretation
    depends on that calibration experiment.

## Results

The result contains:

- **classification:** raw and, when fitted, calibrated class probabilities;
- **redshift:** the marginal posterior, primary and competing basins, and
  coverage-calibrated sets when available; and
- **measured-signal reliability:** a raw score plus a calibrated source
  probability and descriptive grade when available.

Interpret signal reliability using its calibration. Even a narrow redshift
distribution can be unreliable when the measured signal is weak.

Read the [model guide](architecture.md) for the calculations and the
[usage guide](data_and_models.md) for input formats and model files.
