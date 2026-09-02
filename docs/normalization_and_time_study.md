# Flux scaling and time-series study

## Outcome

The current shape and redshift paths are insensitive to a positive rescaling of
each visit. The reported uncertainty scale changes the evidence score, but it
does not change the predicted class or redshift. This is a useful separation:
spectral shape drives inference, while measurement quality drives whether the
result should be trusted.

The visit-change path is useful, but the local pilot has not shown that it uses
the physical direction or rate of evolution. It currently behaves more like a
cross-visit difference encoder than a fully time-aware evolution model.

## Local checks

The checks used the epoch-9 binary model and the flat-redshift calibration
split. Sundial was used only as a transfer check.

### Uncertainty scale

The stored visit scale was multiplied by 0.5, 0.8, 1.0, 1.2, and 2.0 without
changing the spectrum. Across 80 calibration objects:

- no predicted classes changed;
- median changes in `P(Ia)` and predicted redshift were numerical zero;
- Ia F1 stayed at 94.1%;
- Ia median absolute redshift error stayed at 0.133;
- mean evidence changed from 0.82 at 0.5x to 0.55 at 2x;
- residual-only evidence changed from 0.17 at 0.5x to 0.002 at 2x.

The lower-quartile uncertainty stored by preparation is about half the median
uncertainty in both flat-redshift validation and Sundial. This convention is
therefore not interchangeable with a median-error convention when interpreting
the evidence score. RMS uncertainty is unsuitable because detector-edge bins
can dominate it.

In the current 20k prepared set, the object-median scale remains correlated
with redshift (Spearman rho = -0.44 in training and -0.37 in calibration; for
Ia alone rho = -0.58). This is why it must not become an unrestricted class or
redshift channel. Its within-object variation is small (median coefficient of
variation 1.3%), so subtracting one object-wide mean from the log visit
amplitudes largely removes this nuisance while preserving a relative trajectory.

The observer grid has 1024 log-wavelength bins, but the stored Roman-like
spectra have about 201 native detector bins across the same interval. The finer
grid is an interpolation grid, not extra instrumental resolution. Continuum
width is therefore specified in km/s and converted to log-grid bins. A local
150-Ia check rejected very narrow 3000--6000 km/s smoothing as too destructive;
12000 km/s is the current starting value, with 18000 km/s retained as the one
remaining width comparison. Continuum removal is subtraction, never division.

### Observation times

The same 80 objects were evaluated with the full model, the spectral routes
alone, zero dates, reversed visit order, and reassigned cadence patterns.

| Input | Ia F1 | Ia median absolute redshift error |
|---|---:|---:|
| Spectral routes only | 91.4% | 0.322 |
| Full model | 94.1% | 0.133 |
| Dates set to zero | 92.5% | 0.136 |
| Spectra reversed in time | 94.1% | 0.119 |
| Cadence reassigned within redshift | 94.1% | 0.119 |

Residual-only redshift lock stayed near chance in every case. The visit-change
route adds information, but correct time coordinates did not outperform the
controls in this small sample.

### Sundial reliability ranking

On the existing Sundial predictions, the evidence score separates source from
residual spectra with an AUC of 0.92. For true Ia below redshift 2, the fraction
within absolute redshift error 0.1 rises from 33% in the lowest evidence
quintile to 87% in the highest. The score is useful for ranking results, but its
numeric value should not yet be read as a calibrated probability.

## Recommended model contract

1. Keep amplitude-invariant spectral shape and ONIR routes.
2. Keep per-bin uncertainty out of the learned wavelength channels for now.
3. Use a clearly defined robust visit-level uncertainty reference for scaling.
4. Treat the resulting evidence output as a reliability score and calibrate its
   grades on held-out validation data.
5. Keep the current visit-change branch, but only describe it as physical
   temporal evolution after correct dates beat zero, reversed, and reassigned
   controls at scale.
6. Add a simple continuous observed-time or pairwise time-difference comparator
   before considering a larger irregular-time architecture.

## Next comparisons

Use the flat-redshift selection/calibration data to choose between models and
reserve Sundial for transfer testing.

1. Repeat both audits on at least 800 held-out objects on NERSC.
2. Compare the current preprocessing with a continuum-flattened shape view on
   the same 20k split and seed.
3. Compare the current visit-change path with a minimal time-aware path using
   observed time differences and candidate-frame time dilation.
4. Calibrate class probabilities with held-out temperature scaling and
   redshift intervals with stratified conformal calibration.
5. Re-run source, residual, reported-error, visit-count, and cadence controls
   before scaling the selected design.

The current full runs do not need to restart. Their class and redshift paths are
already protected from the visit-level scaling choice. These comparisons are
for the next frozen training recipe and for honest calibration of its outputs.

## Related methods

- SNID motivates a continuum-flattened line-shape comparison:
  <https://arxiv.org/abs/0709.4488>
- SNIascore shows that a recurrent model can classify and estimate redshift from
  low-resolution spectra: <https://arxiv.org/abs/2104.12980>
- ParSNIP is a useful example of physics-aware time-varying spectral modelling,
  although its input is photometry rather than Roman prism spectra:
  <https://arxiv.org/abs/2109.13999>
- mTAN and Neural CDEs are possible irregular-time models if the simple
  time-difference comparator proves useful:
  <https://openreview.net/pdf?id=mXbhcalKnYM> and
  <https://proceedings.neurips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html>
- Temperature scaling and conformal intervals provide separate calibration
  layers without changing the spectral representation:
  <https://proceedings.mlr.press/v70/guo17a.html>

## Reproducibility

The local scripts are:

- `scripts/audit_flux_scale.py`
- `scripts/audit_time_series.py`

The current outputs are in:

- `runs/normalization_audit_epoch9`
- `runs/time_series_audit_epoch9`
