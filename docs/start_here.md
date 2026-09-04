# One object through STRIDER

A STRIDER object is an observation sequence: one or more observer-frame prism
spectra with measured flux, reported uncertainty and observation date.

## Before inference

The fixed reference bank is built once from the training split. Clean simulated
spectra may be placed on a rest-wavelength and broad-phase grid using training
truth during this offline step. The bank records its source split, configuration
digest and edge-weighting semantics.

Training truth is not copied into an inference object. In particular, the model
does not receive the object's true class, true redshift, simulated clean flux or
truth-derived phase.

## Runtime path

1. [`deployment.py`](../src/strider/deployment.py) or
   [`data/dataset.py`](../src/strider/data/dataset.py) validates the measured
   spectra, sorts them chronologically and resamples each visit once onto the
   observer-frame grid.
2. Every visit is divided by one robust uncertainty scale for numerical
   conditioning. Its wavelength-dependent uncertainty is retained for
   deterministic measurement weighting.
3. [`model/coadd.py`](../src/strider/model/coadd.py) reverses the visit scaling
   and forms one inverse-variance accumulated spectrum with propagated error.
4. [`model/roman_reference.py`](../src/strider/model/roman_reference.py) keeps
   the accumulated measured flux unchanged, then forms a normalized full
   spectrum and a continuum-removed view.
5. Both views are aligned and compared with the fixed reference bank at every
   candidate redshift. Fine simulation classes are mapped explicitly to the
   configured reporting classes.
6. At most eight temporal spectra are retained. If more are available, STRIDER
   divides the chronology into equal blocks and selects the strongest measured
   visit in each block, then restores chronological order.
7. Observation intervals are divided by `1 + candidate redshift`. The temporal
   comparison marginalizes over possible starting phases and uses
   uncertainty-weighted relative brightness; it never receives a simulated
   starting phase.
8. Spectral and temporal scores form one joint class-redshift surface.
9. [`model/posterior.py`](../src/strider/model/posterior.py) applies the declared
   prior and redshift-cell widths before normalizing the joint distribution.
10. A separate measured-signal component reports whether the spectra contain
    enough information to use the conditional class-redshift result.

## Public result

The deployment result keeps three meanings separate:

- **classification:** raw and, when fitted, calibrated class probabilities;
- **redshift:** the marginal posterior, primary and competing basins, and
  coverage-calibrated sets when available; and
- **measured-signal reliability:** a raw score plus a calibrated source
  probability and descriptive grade when available.

A narrow redshift posterior is not itself evidence that the measured signal is
sufficient. An insufficient-spectral-information result should therefore remain
distinct from class confidence or posterior shape.

Read [`architecture.md`](architecture.md) for the component boundaries and
[`data_and_models.md`](data_and_models.md) for artifact provenance.
